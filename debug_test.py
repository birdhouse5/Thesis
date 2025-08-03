# debug_test.py
import torch
import numpy as np
import logging
import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.append('.')

def create_mock_data():
    """Create minimal mock data for testing"""
    print("Creating mock data...")
    
    # Create mock dataset file
    import pandas as pd
    
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']  # Just 3 assets for testing
    
    data = []
    for date in dates:
        for ticker in tickers:
            # Mock normalized features (much smaller feature set)
            row = {
                'date': date.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'close_norm': np.random.randn(),
                'volume_norm': np.random.randn(), 
                'returns': np.random.randn() * 0.02,
                'rsi': np.random.uniform(0, 100),
                'market_return_norm': np.random.randn() * 0.01
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv('mock_data.csv', index=False)
    print(f"Mock data created: {df.shape}")
    return 'mock_data.csv'

def test_dataset():
    """Test dataset loading and processing"""
    print("\n=== Testing Dataset ===")
    try:
        from environments.dataset import Dataset
        
        data_path = create_mock_data()
        dataset = Dataset(data_path)
        
        print(f"Dataset loaded: {len(dataset)} days, {dataset.num_assets} assets")
        print(f"Features: {dataset.num_features}")
        print(f"Feature columns: {dataset.feature_cols[:3]}...")  # First 3
        
        # Test window sampling
        window = dataset.get_window(0, 10)
        print(f"Window shape: {window.shape}")  # Should be (10, 3, num_features)
        
        return dataset
        
    except Exception as e:
        print(f"Dataset test FAILED: {e}")
        traceback.print_exc()
        return None

def test_environment(dataset):
    """Test environment with mock dataset"""
    print("\n=== Testing Environment ===")
    try:
        from environments.env import Environment
        
        env = Environment(dataset, episode_length=10, num_assets=3)  # Short episode
        
        # Test reset
        obs = env.reset()
        print(f"Initial obs shape: {obs.shape}")
        
        # Test step with mock action
        mock_action = {
                'decisions': torch.randint(0, 3, (1, 3)),  # (batch=1, 3 assets)
                'long_weights': torch.rand(1, 3),          # (batch=1, 3 assets)  
                'short_weights': torch.rand(1, 3)         # (batch=1, 3 assets)
            }
        
        next_obs, reward, done, info = env.step(mock_action)
        print(f"Step successful: reward={reward}, done={done}")
        print(f"Portfolio weights: {info['portfolio_weights']}")
        
        return env
        
    except Exception as e:
        print(f"Environment test FAILED: {e}")
        traceback.print_exc()
        return None

# In debug_test.py - fix test_models function
def test_models(obs_shape):
    """Test VAE and Policy models"""
    print("\n=== Testing Models ===")
    
    num_assets = obs_shape[0]  # Get actual number of assets (3 in mock data)
    
    # Test VAE
    try:
        from models.vae import VAE
        
        # Use correct action dimension for current number of assets
        action_dim = num_assets * 3  # decisions + long_weights + short_weights
        vae = VAE(obs_dim=obs_shape, action_dim=action_dim, latent_dim=8, hidden_dim=32)
        print("VAE created successfully")
        
        # Test forward pass
        batch_size, seq_len = 1, 5
        mock_obs_seq = torch.randn(batch_size, seq_len, *obs_shape)
        mock_action_seq = torch.randn(batch_size, seq_len, action_dim)  # Use correct dim
        mock_reward_seq = torch.randn(batch_size, seq_len, 1)
        
        latent, mu, logvar, hidden = vae(mock_obs_seq, mock_action_seq, mock_reward_seq)
        print(f"VAE forward pass: latent shape {latent.shape}")
        
    except Exception as e:
        print(f"VAE test FAILED: {e}")
        traceback.print_exc()
    
    # Test Policy
    try:
        from models.policy import Policy
        
        policy = Policy(obs_shape=obs_shape, latent_dim=8, num_assets=num_assets, hidden_dim=32)
        print("Policy created successfully")
        
        # Test forward pass
        mock_obs = torch.randn(1, *obs_shape)
        mock_latent = torch.randn(1, 8)
        
        action, value = policy.act(mock_obs, mock_latent)
        print(f"Policy forward pass: value shape {value.shape}")
        print(f"Action decisions shape: {action['decisions'].shape}")
        
    except Exception as e:
        print(f"Policy test FAILED: {e}")
        traceback.print_exc()

def test_logging():
    """Test logging setup"""
    print("\n=== Testing Logging ===")
    try:
        from logger_config import setup_experiment_logging
        
        exp_logger = setup_experiment_logging("debug_test", log_dir="debug_logs")
        
        # Test scalar logging
        exp_logger.log_scalar('test/metric', 0.5, 0)
        exp_logger.log_hyperparams({'test_param': 42})
        
        print("Logging test successful")
        exp_logger.close()
        
    except Exception as e:
        print(f"Logging test FAILED: {e}")
        traceback.print_exc()

def main():
    """Run all debug tests"""
    print("=== VariBAD Portfolio Debug Test ===")
    
    # Disable GPU for debugging
    torch.cuda.is_available = lambda: False
    
    # Test components step by step
    dataset = test_dataset()
    if dataset is None:
        print("Cannot proceed - Dataset failed")
        return
    
    env = test_environment(dataset)
    if env is None:
        print("Cannot proceed - Environment failed") 
        return
    
    obs_shape = env.reset().shape
    test_models(obs_shape)
    
    test_logging()
    
    print("\n=== Debug Test Complete ===")
    print("If all tests passed, you're ready for GPU training!")

if __name__ == "__main__":
    main()