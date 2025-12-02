import yaml
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_config(config_path: str = "models_config.yaml") -> List[Dict[str, Any]]:
    """Loads the model configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    if not config or "models" not in config:
        raise ValueError("Invalid configuration format. 'models' key is missing.")
        
    return config["models"]

def load_prompt(prompt_path: str = "prompt_instruction.txt") -> str:
    """Loads the system prompt from a text file."""
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def get_api_key(env_var_name: str) -> str:
    """Retrieves an API key from environment variables."""
    key = os.getenv(env_var_name)
    if not key:
        # We don't raise error here immediately to allow other models to run
        # if one key is missing, but we should log a warning in the adapter.
        return "" 
    return key
