#!/usr/bin/env python3
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
            print(f"\n🧪 Running {test_file}...")
            
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
    
    print("\n✅ All tests completed!")
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
