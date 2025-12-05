"""
Metrics calculator for design pattern detection evaluation.
Calculates Precision, Recall, F1-Score per pattern and overall.
FIXED: Counts metrics per FILE, not per detection.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

# Target patterns to evaluate (from the original paper)
TARGET_PATTERNS = ["singleton", "factory", "observer", "decorator"]

# Paper reference results for comparison
PAPER_RESULTS = {
    "ChatGPT": {"precision": 0.768, "recall": 0.857, "f1": 0.810},
    "ChatGPT-Tuned": {"precision": 0.724, "recall": 0.946, "f1": 0.820},
    "Claude": {"precision": 0.752, "recall": 0.762, "f1": 0.757},
    "LLaMA": {"precision": 0.779, "recall": 0.769, "f1": 0.774}
}


def normalize_pattern_name(pattern) -> Optional[str]:
    """
    Normalizes pattern names for comparison.
    Returns None if pattern is not in TARGET_PATTERNS.
    """
    if pattern is None or (isinstance(pattern, float) and pd.isna(pattern)):
        return None
    
    if not isinstance(pattern, str):
        pattern = str(pattern)
    
    pattern = pattern.strip().lower()
    
    if pattern == "none" or pattern == "":
        return None
    
    # Mapping variations to standard names
    mappings = {
        # Factory variations
        'factory method': 'factory',
        'factory pattern': 'factory',
        'abstract factory': 'factory',
        'factory-method': 'factory',
        'factorymethod': 'factory',
        
        # Observer variations
        'observer pattern': 'observer',
        'publisher-subscriber': 'observer',
        'pub-sub': 'observer',
        
        # Strategy variations
        'strategy pattern': 'strategy',
        
        # Singleton variations
        'singleton pattern': 'singleton',
        
        # Decorator variations
        'decorator pattern': 'decorator',
        'wrapper': 'decorator',
    }
    
    normalized = mappings.get(pattern, pattern)
    
    if normalized not in TARGET_PATTERNS:
        return None
    
    return normalized


def load_results_csv(csv_path: str = "results_analysis.csv") -> pd.DataFrame:
    """Loads results from the CSV file."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    
    return pd.read_csv(csv_path)


def calculate_metrics_per_file(
    df: pd.DataFrame, 
    model_name: Optional[str] = None
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int], Dict[str, Any]]:
    """
    Calculates TP, FP, FN per pattern, counting BY FILE (not by detection).
    
    For each file:
    - TP: Expected pattern was correctly detected (among all detections for that file)
    - FN: Expected pattern was NOT detected
    - FP: A pattern was detected in a file where a DIFFERENT pattern was expected
    
    Returns:
        - metrics_per_pattern: Dict with TP, FP, FN counts per pattern
        - ignored_detections: Dict with count of ignored (out-of-scope) patterns
        - debug_info: Additional debugging information
    """
    if model_name:
        df = df[df['model_name'] == model_name]
    
    metrics_per_pattern = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    ignored_detections = defaultdict(int)
    
    # Debug info
    debug_info = {
        "files_processed": 0,
        "correct_detections": [],
        "missed_detections": [],
        "false_positives": []
    }
    
    # Group by filename to process each file once
    grouped = df.groupby('filename')
    
    for filename, file_df in grouped:
        debug_info["files_processed"] += 1
        
        # Get ground truth (should be the same for all rows of this file)
        gt_raw = file_df['ground_truth_label'].iloc[0]
        expected = normalize_pattern_name(gt_raw)
        
        # Skip if ground truth is not a target pattern
        if expected is None:
            continue
        
        # Get all detected patterns for this file (normalized, unique)
        detected_patterns_raw = file_df['detected_pattern'].tolist()
        detected_normalized = set()
        
        for det_raw in detected_patterns_raw:
            det_norm = normalize_pattern_name(det_raw)
            if det_norm is not None:
                detected_normalized.add(det_norm)
            elif det_raw is not None and isinstance(det_raw, str) and det_raw.lower() not in ['none', '']:
                ignored_detections[det_raw] += 1
        
        # Check if expected pattern was detected
        if expected in detected_normalized:
            # TRUE POSITIVE: correctly detected the expected pattern
            metrics_per_pattern[expected]["TP"] += 1
            debug_info["correct_detections"].append({
                "file": filename,
                "expected": expected,
                "detected": list(detected_normalized)
            })
            
            # Check for FALSE POSITIVES: other target patterns detected that weren't expected
            for detected in detected_normalized:
                if detected != expected:
                    metrics_per_pattern[detected]["FP"] += 1
                    debug_info["false_positives"].append({
                        "file": filename,
                        "expected": expected,
                        "wrong_detection": detected
                    })
        else:
            # FALSE NEGATIVE: expected pattern was not detected
            metrics_per_pattern[expected]["FN"] += 1
            debug_info["missed_detections"].append({
                "file": filename,
                "expected": expected,
                "detected": list(detected_normalized) if detected_normalized else ["none"]
            })
            
            # Any target pattern detected is a FALSE POSITIVE
            for detected in detected_normalized:
                metrics_per_pattern[detected]["FP"] += 1
                debug_info["false_positives"].append({
                    "file": filename,
                    "expected": expected,
                    "wrong_detection": detected
                })
    
    return dict(metrics_per_pattern), dict(ignored_detections), debug_info


def calculate_precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Calculates Precision, Recall, and F1-Score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def analyze_model(df: pd.DataFrame, model_name: str, verbose: bool = True) -> Dict[str, Any]:
    """Analyzes results for a specific model."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"ANALYSIS: {model_name.upper()}")
        print(f"{'='*70}")
    
    # Calculate metrics PER FILE
    metrics_per_pattern, ignored_detections, debug_info = calculate_metrics_per_file(df, model_name)
    
    if verbose:
        print(f"\nFiles processed: {debug_info['files_processed']}")
    
    # Show ignored patterns
    if ignored_detections and verbose:
        print(f"\n[!] OUT-OF-SCOPE PATTERNS DETECTED (ignored in metrics):")
        for pattern, count in sorted(ignored_detections.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   - {pattern}: {count}x")
    
    # Display metrics per pattern
    if verbose:
        print(f"\n{'Pattern':<20} {'TP':<6} {'FP':<6} {'FN':<6} {'Prec':<8} {'Recall':<8} {'F1':<8}")
        print("-" * 70)
    
    total_tp, total_fp, total_fn = 0, 0, 0
    pattern_scores = []
    
    for pattern in TARGET_PATTERNS:
        counts = metrics_per_pattern.get(pattern, {"TP": 0, "FP": 0, "FN": 0})
        tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        precision, recall, f1 = calculate_precision_recall_f1(tp, fp, fn)
        pattern_scores.append({
            "pattern": pattern,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
        
        if verbose:
            print(f"{pattern:<20} {tp:<6} {fp:<6} {fn:<6} {precision:<8.3f} {recall:<8.3f} {f1:<8.3f}")
    
    # Overall metrics (micro-average)
    if verbose:
        print("-" * 70)
    overall_precision, overall_recall, overall_f1 = calculate_precision_recall_f1(
        total_tp, total_fp, total_fn
    )
    if verbose:
        print(f"{'OVERALL':<20} {total_tp:<6} {total_fp:<6} {total_fn:<6} {overall_precision:<8.3f} {overall_recall:<8.3f} {overall_f1:<8.3f}")
    
    # Show some debug info
    if verbose and debug_info["missed_detections"]:
        print(f"\n[X] MISSED DETECTIONS (sample of 5):")
        for item in debug_info["missed_detections"][:5]:
            print(f"   - {item['file']}: expected '{item['expected']}', got {item['detected']}")
    
    if verbose and debug_info["false_positives"]:
        print(f"\n[!] FALSE POSITIVES (sample of 5):")
        for item in debug_info["false_positives"][:5]:
            print(f"   - {item['file']}: expected '{item['expected']}', wrongly detected '{item['wrong_detection']}'")
    
    # Qualitative analysis
    model_df = df[df['model_name'] == model_name]
    
    if verbose:
        print(f"\nQUALITATIVE ANALYSIS")
        print("-" * 40)
    
    confidence_scores = pd.to_numeric(model_df['confidence'], errors='coerce').dropna()
    adherence_scores = pd.to_numeric(model_df['adherence'], errors='coerce').dropna()
    
    if len(confidence_scores) > 0 and verbose:
        print(f"  Avg Confidence: {confidence_scores.mean():.1f} (min: {confidence_scores.min():.0f}, max: {confidence_scores.max():.0f})")
    
    if len(adherence_scores) > 0 and verbose:
        print(f"  Avg Adherence:  {adherence_scores.mean():.1f} (min: {adherence_scores.min():.0f}, max: {adherence_scores.max():.0f})")
    
    durations = pd.to_numeric(model_df['duration_seconds'], errors='coerce').dropna()
    if len(durations) > 0 and verbose:
        print(f"  Avg Response Time: {durations.mean():.2f}s (total: {durations.sum():.1f}s)")
    
    # Breakdown by source (DPD vs PMART)
    if verbose:
        print(f"\nBREAKDOWN BY SOURCE")
        print("-" * 40)
        
        for source in ["DPD", "PMART"]:
            source_files = model_df[model_df['filename'].str.startswith(source + "-")]
            if len(source_files) > 0:
                correct = 0
                total = len(source_files['filename'].unique())
                
                for fname in source_files['filename'].unique():
                    file_rows = source_files[source_files['filename'] == fname]
                    gt = normalize_pattern_name(file_rows['ground_truth_label'].iloc[0])
                    detected = set()
                    for det in file_rows['detected_pattern'].tolist():
                        norm = normalize_pattern_name(det)
                        if norm:
                            detected.add(norm)
                    if gt in detected:
                        correct += 1
                
                acc = correct / total if total > 0 else 0
                print(f"   {source:8s}: {correct}/{total} correct ({acc:.1%})")
    
    # Calculate breakdown stats for JSON
    breakdown_stats = {}
    for source in ["DPD", "PMART"]:
        source_files = model_df[model_df['filename'].str.startswith(source + "-")]
        if len(source_files) > 0:
            correct = 0
            total = len(source_files['filename'].unique())
            
            for fname in source_files['filename'].unique():
                file_rows = source_files[source_files['filename'] == fname]
                gt = normalize_pattern_name(file_rows['ground_truth_label'].iloc[0])
                detected = set()
                for det in file_rows['detected_pattern'].tolist():
                    norm = normalize_pattern_name(det)
                    if norm:
                        detected.add(norm)
                if gt in detected:
                    correct += 1
            
            breakdown_stats[source] = {
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total > 0 else 0
            }
    
    return {
        "model": model_name,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "pattern_scores": pattern_scores,
        "ignored_detections": ignored_detections,
        "breakdown_by_source": breakdown_stats,  # NOVO
        "debug_info": {
            "files_processed": debug_info["files_processed"],
            "correct_count": len(debug_info["correct_detections"]),
            "missed_count": len(debug_info["missed_detections"]),
            "fp_count": len(debug_info["false_positives"])
        }
    }


def compare_models(results: List[Dict[str, Any]]) -> None:
    """Compares results across all models."""
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}")
    
    print(f"\n{'Model':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)
    
    sorted_results = sorted(results, key=lambda x: x["f1"], reverse=True)
    
    for result in sorted_results:
        print(f"{result['model']:<25} {result['precision']:<12.3f} {result['recall']:<12.3f} {result['f1']:<12.3f}")
    
    if sorted_results:
        best = sorted_results[0]
        print(f"\n[1st] Best Model: {best['model']} (F1 = {best['f1']:.3f})")


def compare_with_paper() -> None:
    """Shows the original paper results for reference."""
    print(f"\n{'='*70}")
    print("ORIGINAL PAPER RESULTS (IEEE TLT 2025)")
    print(f"{'='*70}")
    
    print(f"\n{'Model':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)
    
    for model, scores in PAPER_RESULTS.items():
        print(f"{model:<25} {scores['precision']:<12.3f} {scores['recall']:<12.3f} {scores['f1']:<12.3f}")
    

def run_metrics(csv_path: str = "results_analysis.csv", save_json: bool = True) -> List[Dict[str, Any]]:
    """Main function to calculate and display all metrics."""
    print("\n" + "="*70)
    print("DESIGN PATTERN DETECTION METRICS (PER-FILE COUNTING)")
    print("="*70)
    
    try:
        df = load_results_csv(csv_path)
    except FileNotFoundError as e:
        print(f"\n[X] Error: {e}")
        print("   Run main.py first to generate results.")
        return []
    
    print(f"\n[OK] Loaded {len(df)} detection rows from {csv_path}")
    
    # Count unique files
    unique_files = df.groupby(['filename', 'model_name']).ngroups
    models = df['model_name'].unique()
    files_per_model = len(df['filename'].unique())
    
    print(f"   Unique files: {files_per_model}")
    print(f"   Models found: {', '.join(models)}")
    print(f"   Note: Multiple detections per file are grouped")
    
    # Analyze each model
    all_results = []
    for model in models:
        result = analyze_model(df, model)
        all_results.append(result)
    
    # Compare models
    if len(all_results) > 1:
        compare_models(all_results)
    
    # Show paper comparison
    compare_with_paper()
    
    # Save metrics to JSON
    if save_json and all_results:
        # Save in same directory as CSV
        csv_dir = Path(csv_path).parent
        output_path = csv_dir / "metrics_summary.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[OK] Metrics saved to: {output_path}")    
    print("\n[OK] Metrics calculation complete!")
    
    return all_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Find most recent results
        results_dir = Path("results")
        if results_dir.exists():
            subdirs = sorted([d for d in results_dir.iterdir() if d.is_dir()], reverse=True)
            if subdirs:
                csv_file = str(subdirs[0] / "results_analysis.csv")
                print(f"Using most recent: {csv_file}")
            else:
                csv_file = "results_analysis.csv"
        else:
            csv_file = "results_analysis.csv"
    
    run_metrics(csv_file)