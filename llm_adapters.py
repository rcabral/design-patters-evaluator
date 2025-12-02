import os
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import requests

# Import SDKs (wrapped in try-except to avoid crashing if not installed during initial setup check)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from config_loader import get_api_key

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMAdapter(ABC):
    """Abstract base class for LLM adapters."""
    
    def __init__(self, config: Dict[str, Any]):
        self.name = config.get("name", "unknown")
        self.model_id = config.get("model_id")
        self.config = config

    @abstractmethod
    def generate(self, system_prompt: str, user_content: str) -> str:
        """Generates a response from the LLM."""
        pass

class OpenAIAdapter(LLMAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        api_key = get_api_key(config.get("api_key_env", ""))
        if not api_key:
            logger.warning(f"API Key for {self.name} not found in environment variables.")
            self.client = None
        elif OpenAI:
            self.client = OpenAI(api_key=api_key)
        else:
            logger.error("OpenAI library not installed.")
            self.client = None

    def generate(self, system_prompt: str, user_content: str) -> str:
        if not self.client:
            return "Error: Client not initialized (missing key or lib)."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2, # Low temperature for more deterministic analysis
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI {self.name}: {e}")
            return f"Error: {str(e)}"

class GoogleGenAIAdapter(LLMAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        api_key = get_api_key(config.get("api_key_env", ""))
        if not api_key:
            logger.warning(f"API Key for {self.name} not found in environment variables.")
            self.model = None
        elif genai:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_id)
        else:
            logger.error("Google Generative AI library not installed.")
            self.model = None

    def generate(self, system_prompt: str, user_content: str) -> str:
        if not self.model:
            return "Error: Model not initialized (missing key or lib)."
        
        try:
            # Gemini often works better with combined prompts or specific system instruction configuration
            # For simplicity and compatibility with 1.5 Pro, we can combine or use system_instruction if supported by the SDK version
            # Here we will concatenate for broad compatibility if system_instruction isn't strictly enforced by the specific model version wrapper
            
            # Newer SDKs support system_instruction in GenerativeModel constructor, but to be safe with dynamic loading:
            full_prompt = f"{system_prompt}\n\nSource Code to Analyze:\n{user_content}"
            
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error calling Google {self.name}: {e}")
            return f"Error: {str(e)}"

class OllamaAdapter(LLMAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.endpoint = config.get("endpoint_url", "http://localhost:11434/api/generate")

    def generate(self, system_prompt: str, user_content: str) -> str:
        # Ollama API structure
        # We can use the /api/chat endpoint for better system prompt support if available, or /api/generate
        # Let's use /api/chat which is more standard for "system" + "user" messages
        
        chat_endpoint = self.endpoint.replace("/api/generate", "/api/chat")
        
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }
        
        try:
            response = requests.post(chat_endpoint, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Error calling Ollama {self.name}: {e}")
            # Fallback to generate if chat fails? No, usually chat is preferred for newer models.
            return f"Error: {str(e)}"

def get_adapter(config: Dict[str, Any]) -> LLMAdapter:
    """Factory function to create the appropriate adapter."""
    provider = config.get("provider", "").lower()
    
    if provider == "openai":
        return OpenAIAdapter(config)
    elif provider == "google":
        return GoogleGenAIAdapter(config)
    elif provider == "ollama":
        return OllamaAdapter(config)
    else:
        raise ValueError(f"Unknown provider: {provider}")
