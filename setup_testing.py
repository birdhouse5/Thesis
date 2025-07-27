#!/usr/bin/env python3
"""
Setup Testing Infrastructure for VariBAD
Creates test files and runs initial baseline capture
"""

import os
import sys
from pathlib import Path


def create_testing_structure():
    """Create the testing directory structure and files"""
    
    # Create test directories
    test_dirs = [
        "tests",
        "tests/data", 
        "tests/baselines",
        "tests/reports"
    ]
    
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")
    
    # Create __init__.py files to make tests a package
    init_files = [
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✓ Created file: {init_file}")
    
    print("\n✅ Test directory structure created!")


def create_baseline_capture_script():
    """Create the baseline capture script in the tests directory"""
    
    baseline_script = Path("tests/baseline_capture.py")
    
    # Copy our baseline capture content
    script_content = '''#!/usr/bin/env python3
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
    print("❌ Could not import test infrastructure")
    print("Make sure test_infrastructure.py is in the tests directory")
    sys.exit(1)


def main():
    """Main baseline capture execution"""
    print("🚀 VariBAD Baseline Capture")
    print("This will capture the current system state before refactoring")
    print("=" * 60)
    
    # Verify we're in the right directory
    if not Path("varibad").exists():
        print("❌ Error: varibad directory not found!")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    # Create baseline capture
    baseline_dir = Path("tests/baselines")
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_capture = BaselineCapture(str(baseline_dir))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\\n1. Checking for existing dataset...")
    
    # Check if dataset exists
    data_path = "data/sp500_rl_ready_cleaned.parquet"
    if os.path.exists(data_path):
        print(f"✅ Found existing dataset: {data_path}")
        
        # Capture dataset baseline
        dataset_baseline = baseline_capture.capture_dataset_baseline(data_path, f"baseline_{timestamp}")
        print(f"✅ Dataset baseline captured")
        
    else:
        print(f"⚠️  Dataset not found at {data_path}")
        print(f"Creating dataset first...")
        
        # Try to create dataset
        try:
            result = subprocess.run([
                sys.executable, 'varibad/main.py', '--mode', 'data_only'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ Dataset created successfully")
                dataset_baseline = baseline_capture.capture_dataset_baseline(data_path, f"baseline_{timestamp}")
            else:
                print(f"❌ Dataset creation failed:")
                print(result.stderr)
                return None
                
        except subprocess.TimeoutExpired:
            print(f"❌ Dataset creation timed out")
            return None
    
    print(f"\\n2. Running quick training test...")
    
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
            print(f"✅ Quick training test passed ({execution_time:.1f}s)")
        else:
            print(f"⚠️  Training test failed (may be expected):")
            print(result.stderr[-500:])  # Last 500 chars
            
    except subprocess.TimeoutExpired:
        print(f"⚠️  Training test timed out")
    
    print(f"\\n3. Creating baseline summary...")
    
    # Create summary file
    summary = f"""VariBAD Baseline Summary
Generated: {timestamp}
========================

Dataset Status: {'✅ Available' if os.path.exists(data_path) else '❌ Missing'}
Baseline ID: {timestamp}

Quick Tests:
- Dataset loading: {'✅ Pass' if os.path.exists(data_path) else '❌ Fail'}
- Training execution: ⚠️  See logs above

Files Created:
- tests/baselines/baseline_{timestamp}_dataset.json

Use this baseline for comparisons during refactoring.
"""
    
    summary_file = Path("tests/baselines/LATEST_BASELINE.txt")
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    print(summary)
    
    print(f"🎯 Baseline capture completed!")
    print(f"Baseline ID: {timestamp}")
    
    return timestamp


if __name__ == "__main__":
    main()
'''
    
    with open(baseline_script, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(baseline_script, 0o755)
    
    print(f"✓ Created baseline capture script: {baseline_script}")


def create_simple_test_runner():
    """Create a simple test runner script"""
    
    runner_script = Path("tests/run_tests.py")
    
    script_content = '''#!/usr/bin/env python3
"""
Simple Test Runner for VariBAD
"""

import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run the test suite"""
    print("🧪 Running VariBAD Test Suite")
    print("=" * 40)
    
    # Check if pytest is available
    try:
        import pytest
        print("✓ pytest is available")
    except ImportError:
        print("❌ pytest not found. Install with: pip install pytest")
        return False
    
    # Run tests
    test_files = [
        "test_phase1_data_pipeline.py"
    ]
    
    for test_file in test_files:
        if Path(f"tests/{test_file}").exists():
            print(f"\\n🧪 Running {test_file}...")
            
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                f"tests/{test_file}", 
                "-v", "--tb=short"
            ])
            
            if result.returncode != 0:
                print(f"❌ Tests failed in {test_file}")
                return False
        else:
            print(f"⚠️  Test file not found: {test_file}")
    
    print("\\n✅ All tests completed!")
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
'''
    
    with open(runner_script, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    os.chmod(runner_script, 0o755)
    print(f"✓ Created test runner: {runner_script}")


def setup_pytest_ini():
    """Create pytest configuration file"""
    
    pytest_ini = Path("pytest.ini")
    
    content = '''[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --color=yes
markers =
    slow: marks tests as slow (may take several minutes)
    integration: marks tests as integration tests  
    baseline: marks tests that establish baselines
'''
    
    with open(pytest_ini, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Created pytest configuration: {pytest_ini}")


def main():
    """Main setup function"""
    print("🛠️  Setting up VariBAD Testing Infrastructure")
    print("=" * 50)
    
    # Check we're in the right place
    if not Path("varibad").exists():
        print("❌ Error: varibad directory not found!")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    print("\\n1. Creating directory structure...")
    create_testing_structure()
    
    print("\\n2. Creating test scripts...")
    create_baseline_capture_script()
    create_simple_test_runner()
    
    print("\\n3. Setting up pytest...")
    setup_pytest_ini()
    
    print("\\n✅ Testing infrastructure setup complete!")
    print("\\nNext steps:")
    print("1. Copy test files to tests/ directory:")
    print("   - test_infrastructure.py")
    print("   - conftest.py") 
    print("   - test_phase1_data_pipeline.py")
    print("2. Run baseline capture: python tests/baseline_capture.py")
    print("3. Run tests: python tests/run_tests.py")


if __name__ == "__main__":
    main()