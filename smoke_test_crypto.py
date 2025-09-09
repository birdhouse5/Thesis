#!/usr/bin/env python3
"""
Simple crypto test with max_episodes=250
"""
import os
from config import generate_experiment_configs, experiment_to_training_config
from main import run_training

def test_single_crypto_experiment():
    """Test a single crypto experiment with 250 episodes"""
    
    # Generate one crypto experiment
    all_experiments = generate_experiment_configs(num_seeds=1)
    crypto_exp = next(exp for exp in all_experiments if exp.asset_class == "crypto" and exp.encoder == "vae")
    
    # Convert to training config
    cfg = experiment_to_training_config(crypto_exp)
    
    # Override for testing
    cfg.max_episodes = 250
    cfg.val_episodes = 10
    cfg.test_episodes = 20
    cfg.val_interval = 50
    cfg.min_episodes_before_stopping = 100
    cfg.early_stopping_patience = 5
    cfg.exp_name = f"{cfg.exp_name}_test250"
    
    print(f"🧪 Testing crypto experiment: {cfg.exp_name}")
    print(f"   - max_episodes: {cfg.max_episodes}")
    print(f"   - val_episodes: {cfg.val_episodes}")
    print(f"   - test_episodes: {cfg.test_episodes}")
    
    # Run the experiment
    results = run_training(cfg)
    
    print(f"\n✅ Test completed!")
    print(f"   - Episodes trained: {results.get('episodes_trained', 'N/A')}")
    print(f"   - Final test reward: {results.get('final_test_reward', 'N/A'):.4f}")
    print(f"   - Success: {results.get('training_completed', False)}")
    
    return results

if __name__ == "__main__":
    # Set up minimal logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Set test environment
    os.environ["DEBUG"] = "true"
    os.environ["TEST_MODE"] = "true"
    
    # Run test
    test_single_crypto_experiment()