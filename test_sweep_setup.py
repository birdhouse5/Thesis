#!/usr/bin/env python3
"""
Test script for VariBAD parameter sweep setup
Validates configuration parsing and model initialization without full training
"""

import json
import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_sweep_config_parsing():
    """Test sweep configuration parsing"""
    print("🧪 Testing sweep configuration parsing...")
    
    from run_experiment import parse_sweep_config
    
    # Test config with sweeps
    test_config = {
        "experiment_name": "test",
        "model": {
            "latent_dim": "SWEEP:[4,6,8]",
            "encoder_hidden": "SWEEP:[64,128]",
            "decoder_hidden": 64
        },
        "training": {
            "num_iterations": 100,
            "vae_updates": "SWEEP:[5,10]"
        }
    }
    
    configs = parse_sweep_config(test_config)
    
    expected_count = 3 * 2 * 2  # 12 combinations
    assert len(configs) == expected_count, f"Expected {expected_count} configs, got {len(configs)}"
    
    # Check first config
    first_config = configs[0]
    assert first_config["model"]["latent_dim"] == 4
    assert first_config["model"]["encoder_hidden"] == 64
    assert first_config["training"]["vae_updates"] == 5
    
    # Check last config
    last_config = configs[-1]
    assert last_config["model"]["latent_dim"] == 8
    assert last_config["model"]["encoder_hidden"] == 128
    assert last_config["training"]["vae_updates"] == 10
    
    print(f"✅ Sweep parsing works: {len(configs)} configurations generated")
    return configs

def test_model_initialization():
    """Test model initialization with different parameters"""
    print("🧪 Testing model initialization...")
    
    from varibad.trainer import VariBADTrainer
    
    # Test different model configurations
    test_configs = [
        {
            "experiment_name": "test_small",
            "training": {"num_iterations": 5, "episode_length": 10, "episodes_per_iteration": 2, "vae_updates": 2},
            "model": {"latent_dim": 4, "encoder_hidden": 64, "decoder_hidden": 64, "policy_hidden": 128},
            "portfolio": {"short_selling": True, "max_short_ratio": 0.3, "transaction_cost": 0.001},
            "learning_rates": {"policy_lr": 0.001, "vae_encoder_lr": 0.001, "vae_decoder_lr": 0.001},
            "environment": {"device": "cpu", "data_path": "data/sp500_rl_ready_cleaned.parquet"}
        },
        {
            "experiment_name": "test_large",
            "training": {"num_iterations": 5, "episode_length": 10, "episodes_per_iteration": 2, "vae_updates": 2},
            "model": {"latent_dim": 12, "encoder_hidden": 256, "decoder_hidden": 128, "policy_hidden": 512},
            "portfolio": {"short_selling": False, "max_short_ratio": 0.0, "transaction_cost": 0.001},
            "learning_rates": {"policy_lr": 0.0001, "vae_encoder_lr": 0.0001, "vae_decoder_lr": 0.0001},
            "environment": {"device": "cpu", "data_path": "data/sp500_rl_ready_cleaned.parquet"}
        }
    ]
    
    for i, config in enumerate(test_configs):
        print(f"  Testing configuration {i+1}...")
        
        try:
            trainer = VariBADTrainer(config)
            param_count = trainer.get_parameter_count()
            device = trainer.device
            
            print(f"    ✓ Model initialized: {param_count:,} parameters on {device}")
            
            # Test single episode collection
            episode_stats = trainer.collect_episode(deterministic=True)
            print(f"    ✓ Episode collection: {episode_stats['steps']} steps, reward {episode_stats['total_reward']:.4f}")
            
            # Test VAE training (single step)
            if trainer.buffer.get_buffer_stats()['num_sequences'] > 0:
                vae_stats = trainer.train_vae(batch_size=1, max_seq_length=5)
                print(f"    ✓ VAE training: loss {vae_stats.get('vae_loss', 0):.4f}")
            
        except Exception as e:
            print(f"    ❌ Configuration {i+1} failed: {e}")
            raise
    
    print("✅ Model initialization works for all configurations")

def test_data_availability():
    """Test data availability"""
    print("🧪 Testing data availability...")
    
    from varibad.data import load_dataset
    
    data_path = "data/sp500_rl_ready_cleaned.parquet"
    
    try:
        # Just try to load the dataset directly
        df = load_dataset(data_path)
        
        print(f"  ✓ Dataset loaded: {df.shape} shape")
        print(f"  ✓ Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"  ✓ Tickers: {df['ticker'].nunique()}")
        
        # Check for required columns
        required_cols = ['date', 'ticker', 'returns', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        assert len(missing_cols) == 0, f"Missing columns: {missing_cols}"
        
        # Check data size
        assert len(df) > 1000, "Dataset too small"
        
        print("✅ Data pipeline works correctly")
        
    except Exception as e:
        print(f"  ⚠️  Dataset issue: {e}")
        print("  Creating dataset...")
        
        from varibad.data import create_dataset
        create_dataset(data_path)
        
        df = load_dataset(data_path)
        print(f"  ✓ Dataset created: {df.shape} shape")
        print("✅ Data pipeline works correctly")

def test_environment_creation():
    """Test environment creation"""
    print("🧪 Testing environment creation...")
    
    from varibad.data import load_dataset
    from varibad.models import PortfolioEnvironment
    
    data = load_dataset("data/sp500_rl_ready_cleaned.parquet")
    
    # Test different environment configurations
    env_configs = [
        {"episode_length": 10, "enable_short_selling": True, "max_short_ratio": 0.3},
        {"episode_length": 30, "enable_short_selling": False, "max_short_ratio": 0.0},
    ]
    
    for i, config in enumerate(env_configs):
        print(f"  Testing environment {i+1}...")
        
        env = PortfolioEnvironment(data, **config)
        
        # Test reset
        state = env.reset()
        print(f"    ✓ State shape: {state.shape}")
        print(f"    ✓ Action space: {env.action_space.shape}")
        
        # Test step
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        
        print(f"    ✓ Step works: reward {reward:.4f}, done {done}")
        print(f"    ✓ Info keys: {list(info.keys())}")
    
    print("✅ Environment creation works correctly")

def test_config_files():
    """Test existing config files"""
    print("🧪 Testing config files...")
    
    config_dir = Path("config")
    config_files = list(config_dir.glob("*.conf"))
    
    if not config_files:
        print("  ⚠️  No config files found, creating test configs...")
        
        # Create basic test config
        test_config = {
            "experiment_name": "test_config",
            "training": {"num_iterations": 10, "episode_length": 5, "episodes_per_iteration": 2, "vae_updates": 2},
            "model": {"latent_dim": 4, "encoder_hidden": 64, "decoder_hidden": 64, "policy_hidden": 128},
            "portfolio": {"short_selling": True, "max_short_ratio": 0.3, "transaction_cost": 0.001},
            "learning_rates": {"policy_lr": 0.001, "vae_encoder_lr": 0.001, "vae_decoder_lr": 0.001},
            "environment": {"device": "auto", "data_path": "data/sp500_rl_ready_cleaned.parquet"}
        }
        
        config_dir.mkdir(exist_ok=True)
        with open(config_dir / "test.conf", 'w') as f:
            json.dump(test_config, f, indent=2)
        
        config_files = [config_dir / "test.conf"]
    
    for config_file in config_files[:2]:  # Test first 2 configs only
        print(f"  Testing {config_file.name}...")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Basic validation
        required_sections = ['training', 'model', 'portfolio', 'learning_rates', 'environment']
        for section in required_sections:
            assert section in config, f"Missing section: {section}"
        
        print(f"    ✓ Config structure valid")
    
    print("✅ Config files are valid")

def main():
    """Run all tests"""
    print("🚀 Running VariBAD sweep setup tests\n")
    
    tests = [
        test_data_availability,
        test_environment_creation,
        test_config_files,
        test_sweep_config_parsing,
        test_model_initialization,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
            print()
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            failed += 1
            print()
    
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Sweep setup is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)