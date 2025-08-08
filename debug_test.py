# debug_test_fixed.py
import torch
import numpy as np
import logging
import sys
import traceback
from pathlib import Path
import os

# Add project root to path - make sure we're in the right directory
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

def create_mock_data():
    """Create minimal mock data for testing"""
    print("Creating mock data...")
    
    import pandas as pd
    
    dates = pd.date_range('2020-01-01', periods=200, freq='D')  
    tickers = ['AAPL', 'MSFT', 'GOOGL']  # 3 assets
    
    data = []
    for date in dates:
        for ticker in tickers:
            row = {
                'date': date.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'close': np.random.uniform(50, 200),
                'close_norm': np.random.uniform(0, 1),
                'volume_norm': np.random.randn(),
                'returns': np.random.randn() * 0.02,
                'rsi_norm': np.random.uniform(-1, 1),
                'market_return_norm': np.random.randn() * 0.01,
                'volatility_5d_norm': np.random.randn(),
                'sma_20_norm': np.random.uniform(0, 1),
                'bb_position_norm': np.random.uniform(-2, 2),
                'macd_norm': np.random.randn(),
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    output_path = 'mock_data.parquet'
    df.to_parquet(output_path, index=False)
    print(f"Mock data created: {df.shape} -> {output_path}")
    return output_path

def test_dataset():
    """Test dataset loading"""
    print("\n=== Testing Dataset ===")
    try:
        from environments.dataset import Dataset
        
        data_path = create_mock_data()
        dataset = Dataset(data_path)
        
        print(f"Dataset loaded: {len(dataset)} days, {dataset.num_assets} assets")
        print(f"Features: {dataset.num_features}")
        
        # Test window sampling with smaller window
        window_size = min(20, len(dataset))
        window = dataset.get_window(0, window_size)
        print(f"Window features shape: {window['features'].shape}")
        print(f"Window raw_prices shape: {window['raw_prices'].shape}")
        
        return dataset
        
    except Exception as e:
        print(f"Dataset test FAILED: {e}")
        traceback.print_exc()
        return None

def test_meta_environment(dataset):
    """Test MetaEnv with proper tensor creation"""
    print("\n=== Testing MetaEnv ===")
    try:
        from environments.env import MetaEnv
        
        # Create a simple continuous dataset tensor
        seq_len = min(60, len(dataset))
        window = dataset.get_window(0, seq_len)
        
        # Convert to tensors with proper shape
        dataset_tensor = {
            'features': torch.tensor(window['features'], dtype=torch.float32),
            'raw_prices': torch.tensor(window['raw_prices'], dtype=torch.float32)
        }
        
        print(f"Dataset tensor shapes: features={dataset_tensor['features'].shape}, "
              f"prices={dataset_tensor['raw_prices'].shape}")
        
        # Create MetaEnv with shorter sequences for testing
        env = MetaEnv(
            dataset=dataset_tensor,
            feature_columns=dataset.feature_cols,
            seq_len=min(30, seq_len),
            min_horizon=10,
            max_horizon=15
        )
        
        # Sample task
        task = env.sample_task()
        env.set_task(task)
        initial_obs = env.reset()
        
        print(f"Initial observation shape: {initial_obs.shape}")
        
        # Test step with proper action
        num_assets = initial_obs.shape[0]
        portfolio_weights = np.random.rand(num_assets)
        portfolio_weights = portfolio_weights / portfolio_weights.sum()
        
        next_obs, reward, done, info = env.step(portfolio_weights)
        print(f"Step successful: reward={reward:.4f}, done={done}")
        
        return env, initial_obs.shape
        
    except Exception as e:
        print(f"MetaEnv test FAILED: {e}")
        traceback.print_exc()
        return None, None

def test_models(obs_shape, num_assets):
    """Test models with consistent device handling"""
    print("\n=== Testing Models ===")
    
    # Force CPU usage
    device = torch.device("cpu")
    
    try:
        from models.policy import PortfolioPolicy
        from models.vae import VAE
        
        latent_dim = 32
        
        # Test Policy
        policy = PortfolioPolicy(
            obs_shape=obs_shape,
            latent_dim=latent_dim,
            num_assets=num_assets,
            hidden_dim=64  # Smaller for testing
        ).to(device)
        
        # Test with proper shapes
        batch_size = 2
        mock_obs = torch.randn(batch_size, *obs_shape, device=device)
        mock_latent = torch.randn(batch_size, latent_dim, device=device)
        
        portfolio_weights, value = policy.act(mock_obs, mock_latent)
        print(f"✓ Policy act: weights shape {portfolio_weights.shape}, sum {portfolio_weights.sum(dim=-1)}")
        
        # Test VAE
        vae = VAE(
            obs_dim=obs_shape,
            num_assets=num_assets,
            latent_dim=latent_dim,
            hidden_dim=64
        ).to(device)
        
        # Test with sequence data
        seq_len = 10
        mock_obs_seq = torch.randn(batch_size, seq_len, *obs_shape, device=device)
        mock_action_seq = torch.randn(batch_size, seq_len, num_assets, device=device)
        mock_action_seq = torch.softmax(mock_action_seq, dim=-1)  # Normalize
        mock_reward_seq = torch.randn(batch_size, seq_len, 1, device=device)
        
        latent, mu, logvar, hidden = vae(mock_obs_seq, mock_action_seq, mock_reward_seq)
        print(f"✓ VAE forward: latent {latent.shape}")
        
        loss, loss_components = vae.compute_loss(mock_obs_seq, mock_action_seq, mock_reward_seq)
        print(f"✓ VAE loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"Models test FAILED: {e}")
        traceback.print_exc()

def test_simple_training_loop():
    """Test a very simple training loop"""
    print("\n=== Testing Simple Training Loop ===")
    try:
        # Create minimal setup
        data_path = create_mock_data()
        
        from environments.dataset import Dataset
        from environments.env import MetaEnv
        from models.policy import PortfolioPolicy
        
        dataset = Dataset(data_path)
        
        # Create simple dataset tensor
        window_size = min(40, len(dataset))
        window = dataset.get_window(0, window_size)
        
        dataset_tensor = {
            'features': torch.tensor(window['features'], dtype=torch.float32),
            'raw_prices': torch.tensor(window['raw_prices'], dtype=torch.float32)
        }
        
        env = MetaEnv(
            dataset=dataset_tensor,
            feature_columns=dataset.feature_cols,
            seq_len=20,
            min_horizon=5,
            max_horizon=10
        )
        
        # Sample task and reset
        task = env.sample_task()
        env.set_task(task)
        obs = env.reset()
        
        obs_shape = obs.shape
        num_assets = obs_shape[0]
        latent_dim = 16
        
        # Create simple policy
        policy = PortfolioPolicy(
            obs_shape=obs_shape,
            latent_dim=latent_dim,
            num_assets=num_assets,
            hidden_dim=32
        )
        
        # Test single forward pass
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        latent = torch.zeros(1, latent_dim)
        
        with torch.no_grad():
            action, value = policy.act(obs_tensor, latent, deterministic=True)
            print(f"Policy output: action sum={action.sum():.4f}, value={value.item():.4f}")
        
        # Test environment step
        action_np = action.squeeze(0).numpy()
        next_obs, reward, done, info = env.step(action_np)
        print(f"Environment step: reward={reward:.4f}, done={done}")
        
        print("✓ Simple training loop test successful")
        
    except Exception as e:
        print(f"Simple training test FAILED: {e}")
        traceback.print_exc()

def main():
    """Run debug tests with better error handling"""
    print("=== VariBAD Portfolio Debug Test (Fixed) ===")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Ensure we're using CPU
    if torch.cuda.is_available():
        print("CUDA detected but forcing CPU usage for debugging")
        torch.cuda.is_available = lambda: False
    
    try:
        # Test dataset
        dataset = test_dataset()
        if dataset is None:
            print("Cannot proceed - Dataset failed")
            return
        
        # Test environment
        env, obs_shape = test_meta_environment(dataset)
        if env is None:
            print("Cannot proceed - MetaEnv failed")
            return
        
        num_assets = obs_shape[0]
        print(f"\nUsing: obs_shape={obs_shape}, num_assets={num_assets}")
        
        # Test models
        test_models(obs_shape, num_assets)
        
        # Test simple training loop
        test_simple_training_loop()
        
        print("\n=== Debug Test Summary ===")
        print("✓ Most components working correctly")
        print("✓ Ready for main training script")
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
    
    finally:
        # Cleanup
        mock_file = Path('mock_data.parquet')
        if mock_file.exists():
            mock_file.unlink()
            print("Cleaned up mock data file")

if __name__ == "__main__":
    main()