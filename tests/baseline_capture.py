#!/usr/bin/env python3
"""
Baseline Capture Script for VariBAD Portfolio Optimization
Captures the current system state before refactoring begins
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np



# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the baseline capture class from our test infrastructure
try:
    from test_infrastructure import BaselineCapture
except ImportError:
    print("[FAIL] Could not import test infrastructure")
    print("Make sure test_infrastructure.py is in the tests directory")
    sys.exit(1)


def main():
    """Main baseline capture execution"""
    print("🚀 VariBAD Baseline Capture")
    print("This will capture the current system state before refactoring")
    print("=" * 60)
    
    # Verify we're in the right directory
    if not Path("varibad").exists():
        print("[FAIL] Error: varibad directory not found!")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    # Create baseline capture
    baseline_dir = Path("tests/baselines")
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_capture = BaselineCapture(str(baseline_dir))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n1. Checking for existing dataset...")
    
    # Check if dataset exists
    data_path = "varibad/data/sp500_rl_ready_cleaned.parquet"
    if os.path.exists(data_path):
        print(f"[OK] Found existing dataset: {data_path}")
        
        # Capture dataset baseline
        dataset_baseline = baseline_capture.capture_dataset_baseline(data_path, f"baseline_{timestamp}")
        print(f"[OK] Dataset baseline captured")
        
    else:
        print(f"[WARN]  Dataset not found at {data_path}")
        print(f"Creating dataset first...")
        
        # Try to create dataset
        try:
            result = subprocess.run([
                sys.executable, 'varibad/main.py', '--mode', 'data_only'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"[OK] Dataset created successfully")
                dataset_baseline = baseline_capture.capture_dataset_baseline(data_path, f"baseline_{timestamp}")
            else:
                print(f"[FAIL] Dataset creation failed:")
                print(result.stderr)
                return None
                
        except subprocess.TimeoutExpired:
            print(f"[FAIL] Dataset creation timed out")
            return None
    
    print(f"\n2. Running quick training test...")
    
    # Quick training test for baseline
    try:
        start_time = time.time()
        result = subprocess.run([
            sys.executable, 'varibad/main.py',
            '--mode', 'train',
            '--num_iterations', '3',
            '--episode_length', '5', 
            '--episodes_per_iteration', '2',
            '--vae_updates', '2'
        ], capture_output=True, text=True, timeout=300)
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[OK] Quick training test passed ({execution_time:.1f}s)")
        else:
            print(f"[WARN]  Training test failed (may be expected):")
            print(result.stderr[-500:])  # Last 500 chars
            
    except subprocess.TimeoutExpired:
        print(f"[WARN]  Training test timed out")
    
    print(f"\n3. Creating baseline summary...")
    
    # Create summary file
    summary = f"""VariBAD Baseline Summary
Generated: {timestamp}
========================

Dataset Status: {'[OK] Available' if os.path.exists(data_path) else '[FAIL] Missing'}
Baseline ID: {timestamp}

Quick Tests:
- Dataset loading: {'[OK] Pass' if os.path.exists(data_path) else '[FAIL] Fail'}
- Training execution: [WARN]  See logs above

Files Created:
- tests/baselines/baseline_{timestamp}_dataset.json

Use this baseline for comparisons during refactoring.
"""
    
    summary_file = Path("tests/baselines/LATEST_BASELINE.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(summary)
    
    print(f"🎯 Baseline capture completed!")
    print(f"Baseline ID: {timestamp}")
    
    return timestamp


if __name__ == "__main__":
    main()
