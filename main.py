import os
import json
import re
import pandas as pd
import logging
from typing import List, Dict, Any
from datetime import datetime

from config_loader import load_config, load_prompt
from llm_adapters import get_adapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

RESULTS_FILE = "results_analysis.csv"
SOURCE_DIR = "design-patterns"

def extract_json(text: str) -> Dict[str, Any]:
    """Extracts JSON object from a string using regex."""
    try:
        # First try to parse the whole text
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a code block with json
    json_block_pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(json_block_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find just the first curly brace to the last curly brace
    brace_pattern = r"(\{.*\})"
    match = re.search(brace_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
            
    return {}

def scan_files(root_dir: str) -> List[Dict[str, str]]:
    """Scans the directory for source files and determines ground truth."""
    files_to_process = []
    
    if not os.path.exists(root_dir):
        logger.error(f"Source directory '{root_dir}' not found.")
        return []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.java', '.py', '.cpp', '.cs', '.js', '.ts')): # Add extensions as needed
                full_path = os.path.join(root, file)
                # Ground truth is the name of the immediate parent folder
                ground_truth = os.path.basename(root)
                files_to_process.append({
                    "path": full_path,
                    "filename": file,
                    "ground_truth": ground_truth
                })
    
    return files_to_process

def main():
    logger.info("Starting Design Pattern Evaluator...")
    
    # 1. Load Configuration
    try:
        models_conf = load_config()
        system_prompt = load_prompt()
    except Exception as e:
        logger.critical(f"Configuration error: {e}")
        return

    # 2. Initialize Adapters
    adapters = []
    for m_conf in models_conf:
        try:
            adapter = get_adapter(m_conf)
            adapters.append(adapter)
            logger.info(f"Loaded adapter for: {adapter.name}")
        except Exception as e:
            logger.error(f"Failed to load adapter for {m_conf.get('name')}: {e}")

    if not adapters:
        logger.critical("No valid adapters loaded. Exiting.")
        return

    # 3. Scan Files
    files = scan_files(SOURCE_DIR)
    logger.info(f"Found {len(files)} files to process.")
    
    if not files:
        logger.warning(f"No files found in {SOURCE_DIR}. Please ensure the directory exists and contains code.")
        # Create the directory if it doesn't exist to help the user
        if not os.path.exists(SOURCE_DIR):
            os.makedirs(SOURCE_DIR)
            logger.info(f"Created empty directory '{SOURCE_DIR}'. Please populate it with source code.")
        return

    results = []

    # 4. Batch Execution
    for i, file_info in enumerate(files):
        logger.info(f"Processing file {i+1}/{len(files)}: {file_info['filename']} (True: {file_info['ground_truth']})")
        
        try:
            with open(file_info['path'], 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_info['path']}: {e}")
            continue

        for adapter in adapters:
            logger.info(f"  Querying {adapter.name}...")
            start_time = datetime.now()
            
            raw_response = adapter.generate(system_prompt, source_code)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Parse Response
            parsed_json = extract_json(raw_response)
            
            # Extract fields (handling potential list inside the JSON as per prompt instruction)
            # The prompt asks for: "design_patterns": [{ ... }]
            patterns_found = parsed_json.get("design_patterns", [])
            
            # If list is empty or not found, we record a "None" detection
            if not patterns_found:
                results.append({
                    "filename": file_info['filename'],
                    "ground_truth_label": file_info['ground_truth'],
                    "model_name": adapter.name,
                    "detected_pattern": "None",
                    "confidence": 0,
                    "adherence": 0,
                    "reasoning": "No pattern detected or parse error",
                    "raw_response": raw_response,
                    "duration_seconds": duration
                })
            else:
                for p in patterns_found:
                    results.append({
                        "filename": file_info['filename'],
                        "ground_truth_label": file_info['ground_truth'],
                        "model_name": adapter.name,
                        "detected_pattern": p.get("pattern", "Unknown"),
                        "confidence": p.get("confidence", 0),
                        "adherence": p.get("adherence", 0),
                        "reasoning": p.get("reason", ""),
                        "raw_response": raw_response, # Storing raw response for debug
                        "duration_seconds": duration
                    })

    # 5. Save Results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(RESULTS_FILE, index=False)
        logger.info(f"Analysis complete. Results saved to {RESULTS_FILE}")
    else:
        logger.info("No results generated.")

if __name__ == "__main__":
    main()
