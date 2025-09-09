#!/usr/bin/env python3
"""
Test script to specifically run crypto experiments
"""
import os
import shutil
from pathlib import Path

# Import your modules
from config import generate_experiment_configs
from experiment_manager import ExperimentManager

def test_crypto_experiments():
    """Test crypto dataset experiments specifically"""
    
    # Clear any existing checkpoints to force fresh run
    checkpoint_dir = Path("experiment_checkpoints")
    if checkpoint_dir.exists():
        print("🧹 Clearing previous checkpoints...")
        shutil.rmtree(checkpoint_dir)
        checkpoint_dir.mkdir()
    
    # Generate all experiments
    all_experiments = generate_experiment_configs(num_seeds=2)  # Just 2 seeds for testing
    
    # Filter for crypto experiments only
    crypto_experiments = [exp for exp in all_experiments if exp.asset_class == "crypto"]
    
    print(f"🚀 Testing {len(crypto_experiments)} crypto experiments:")
    for exp in crypto_experiments:
        print(f"   - {exp.asset_class}_{exp.encoder}_seed{exp.seed}")
    
    # Create experiment manager
    manager = ExperimentManager(crypto_experiments, max_retries=0)
    
    # Run crypto experiments
    print("\n📊 Starting crypto experiment batch...")
    summary = manager.run_all_experiments("crypto_test_study")
    
    # Print results
    print("\n" + "="*60)
    print("CRYPTO TEST RESULTS")
    print("="*60)
    
    batch_info = summary.get('batch_info', {})
    print(f"Total experiments: {batch_info.get('total_experiments', 0)}")
    print(f"Completed: {batch_info.get('completed', 0)}")
    print(f"Failed: {batch_info.get('failed', 0)}")
    print(f"Success rate: {batch_info.get('success_rate', 0)*100:.1f}%")
    
    if 'performance_stats' in summary and summary['performance_stats']:
        perf = summary['performance_stats']
        print(f"\nPerformance Summary:")
        print(f"Average test reward: {perf.get('avg_test_reward', 0):.4f}")
        print(f"Best test reward: {perf.get('best_test_reward', 0):.4f}")
        print(f"Average Sharpe: {perf.get('avg_sharpe', 0):.4f}")
        print(f"Best Sharpe: {perf.get('best_sharpe', 0):.4f}")
    
    if 'error_analysis' in summary and summary['error_analysis']:
        print(f"\nErrors encountered:")
        for error_type, count in summary['error_analysis'].items():
            print(f"  {error_type}: {count}")
    
    return summary

if __name__ == "__main__":
    # Set up logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set environment for debugging
    os.environ["DEBUG"] = "true"
    
    # Run crypto tests
    summary = test_crypto_experiments()