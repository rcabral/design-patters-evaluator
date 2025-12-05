import os
import json
import re
import pandas as pd
import logging
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
import shutil

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

SOURCE_DIR = "design-patterns"

def get_results_dir() -> Path:
    """Creates timestamped results directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = Path("results") / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

RESULTS_FILE = "results_analysis.csv"

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
            if file.endswith(('.java', '.py', '.cpp', '.cs', '.js', '.ts')):
                full_path = os.path.join(root, file)
                # Ground truth is the name of the immediate parent folder
                ground_truth = os.path.basename(root)
                files_to_process.append({
                    "path": full_path,
                    "filename": file,
                    "ground_truth": ground_truth
                })
    
    return files_to_process

def run_evaluation() -> bool:
    """Runs the LLM evaluation pipeline."""
    results_dir = get_results_dir()
    
    logger.info(f"Starting Design Pattern Evaluator...")
    logger.info(f"Source: {SOURCE_DIR}")
    logger.info(f"Results: {results_dir}")
    
    # 1. Load Configuration
    try:
        models_conf = load_config()
        system_prompt = load_prompt()
    except Exception as e:
        logger.critical(f"Configuration error: {e}")
        return False

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
        return False

    # 3. Scan Files
    files = scan_files(SOURCE_DIR)
    logger.info(f"Found {len(files)} files to process.")
    
    if not files:
        logger.warning(f"No files found in {SOURCE_DIR}.")
        return False

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
            patterns_found = parsed_json.get("design_patterns", [])
            
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
                        "raw_response": raw_response,
                        "duration_seconds": duration
                    })

    # 5. Save Results
    if not results:
        return False
    
    df = pd.DataFrame(results)
    
    # Save combined results
    combined_file = results_dir / "results_analysis.csv"
    df.to_csv(combined_file, index=False)
    logger.info(f"Combined results saved to {combined_file}")
    
    # 6. Separate by source (DPD / PMART)
    for source in ["DPD", "PMART"]:
        source_df = df[df['filename'].str.startswith(source + "-")]
        if len(source_df) > 0:
            source_dir = results_dir / source.lower()
            source_dir.mkdir(exist_ok=True)
            
            source_file = source_dir / "results_analysis.csv"
            source_df.to_csv(source_file, index=False)
            logger.info(f"{source} results saved to {source_file}")
    
    # 7. Save metadata
    metadata = {
        "source_dir": SOURCE_DIR,
        "timestamp": datetime.now().isoformat(),
        "files_processed": len(files),
        "models_used": [a.name for a in adapters],
        "total_results": len(results),
        "dpd_files": len(df[df['filename'].str.startswith("DPD-")]['filename'].unique()),
        "pmart_files": len(df[df['filename'].str.startswith("PMART-")]['filename'].unique())
    }
    with open(results_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 8. Run metrics for each
    print("\n" + "="*70)
    print("CALCULATING METRICS...")
    print("="*70)
    
    from metrics import run_metrics
    
    # Combined metrics
    print("\n>> COMBINED RESULTS:")
    run_metrics(str(combined_file))
    
    # Per-source metrics
    for source in ["dpd", "pmart"]:
        source_file = results_dir / source / "results_analysis.csv"
        if source_file.exists():
            print(f"\n>> {source.upper()} RESULTS:")
            run_metrics(str(source_file))
    
    return True


def main():
    """Simple entry point."""
    if not Path(SOURCE_DIR).exists():
        print(f"Source directory '{SOURCE_DIR}' not found.")
        print(f"Run: python datasets-raw/generate_datasets.py")
        return
    
    run_evaluation()


if __name__ == "__main__":
    main()