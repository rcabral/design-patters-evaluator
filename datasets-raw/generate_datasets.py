"""
Generates design-patterns/ folder from DPD and P-MARt datasets.
Simple execution: python generate_datasets.py

Files are prefixed with origin: DPD- or PMART-
"""

import csv
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Configuration
SEED = 42
SAMPLES_PER_PATTERN = 10
TARGET_PATTERNS = ["singleton", "factory", "observer", "decorator"]

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT = SCRIPT_DIR.parent / "design-patterns"

# DPD paths
DPD_DIR = SCRIPT_DIR / "Java-DPD-dataset" / "dataset"
DPD_CSV = DPD_DIR / "final_dataset.csv"

# P-MARt paths
PMART_SOURCE = SCRIPT_DIR / "ptidej-P-MARt"


def main():
    print("\n" + "="*60)
    print("DESIGN PATTERN DATASET GENERATOR")
    print("="*60)
    print(f"   Patterns: {', '.join(TARGET_PATTERNS)}")
    print(f"   Samples per pattern per source: {SAMPLES_PER_PATTERN}")
    
    random.seed(SEED)
    
    # Clear output
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    
    stats = {"DPD": defaultdict(int), "PMART": defaultdict(int)}
    
    # === DPD ===
    print(f"\n[DPD] Loading...")
    if DPD_CSV.exists():
        data = []
        with open(DPD_CSV, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                data.append(row)
        
        label_map = {"Factory": "factory", "Singleton": "singleton", 
                     "Observer": "observer", "Decorator": "decorator"}
        
        patterns = defaultdict(list)
        for row in data:
            norm = label_map.get(row.get('Label', ''))
            if norm:
                patterns[norm].append(row)
        
        for pattern in TARGET_PATTERNS:
            items = patterns.get(pattern, [])
            if not items:
                continue
            
            pattern_dir = OUTPUT / pattern
            pattern_dir.mkdir(exist_ok=True)
            
            sample = random.sample(items, min(SAMPLES_PER_PATTERN, len(items)))
            for row in sample:
                filename = f"{row['Project']}-{row['Class']}.java"
                src = DPD_DIR / filename
                if src.exists():
                    dst = pattern_dir / f"DPD-{filename}"
                    shutil.copy2(src, dst)
                    stats["DPD"][pattern] += 1
    else:
        print(f"   [!] DPD CSV not found")
    
    # === P-MARt ===
    print(f"[PMART] Loading...")
    if PMART_SOURCE.exists():
        keywords = {
            'singleton': ['singleton'],
            'factory': ['factory'],
            'observer': ['observer', 'listener'],
            'decorator': ['decorator', 'wrapper']
        }
        
        patterns_found = defaultdict(list)
        for java_file in PMART_SOURCE.rglob("*.java"):
            file_lower = java_file.stem.lower()
            for pattern, kws in keywords.items():
                if any(kw in file_lower for kw in kws):
                    project = "PMARt"
                    for part in java_file.parts:
                        if any(part.startswith(p) for p in ['Ant', 'ArgoUML', 'Azureus', 
                                'JHotDraw', 'DrJava', 'Eclipse', 'Xerces', 'Xalan']):
                            project = part.split()[0].replace(' ', '-')
                            break
                    patterns_found[pattern].append((java_file, project))
                    break
        
        for pattern in TARGET_PATTERNS:
            items = patterns_found.get(pattern, [])
            if not items:
                continue
            
            pattern_dir = OUTPUT / pattern
            pattern_dir.mkdir(exist_ok=True)
            
            random.shuffle(items)
            seen = set()
            count = 0
            
            for path, project in items:
                if count >= SAMPLES_PER_PATTERN:
                    break
                dst_name = f"PMART-{project}-{path.stem}.java"
                if dst_name not in seen:
                    seen.add(dst_name)
                    dst = pattern_dir / dst_name
                    if not dst.exists():
                        shutil.copy2(path, dst)
                        stats["PMART"][pattern] += 1
                        count += 1
    else:
        print(f"   [!] P-MARt source not found")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Pattern':<15} {'DPD':<8} {'PMART':<8} {'Total':<8}")
    print("-" * 40)
    
    total_dpd, total_pmart = 0, 0
    for p in TARGET_PATTERNS:
        d, m = stats["DPD"][p], stats["PMART"][p]
        total_dpd += d
        total_pmart += m
        print(f"{p:<15} {d:<8} {m:<8} {d+m:<8}")
    
    print("-" * 40)
    print(f"{'TOTAL':<15} {total_dpd:<8} {total_pmart:<8} {total_dpd+total_pmart:<8}")
    print(f"\n[OK] Output: {OUTPUT}")
    print(f"[OK] Run 'python main.py' to evaluate.")


if __name__ == "__main__":
    main()