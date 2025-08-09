# debug_test_splits.py - Updated to test train-test-val split functionality
import torch
import numpy as np
import logging
import sys
import traceback
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

def create_mock_data():
    """Create minimal mock data with multiple years for testing splits"""
    print("Creating mock data with temporal range...")
    
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create 6 years of data: 2013-2018 for testing splits
    start_date = datetime(2013, 1, 1)
    end_date = datetime(2018, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    tickers = ['AAPL', 'MSFT', 'GOOGL']  # 3 assets
    
    data = []
    for date in dates:
        for ticker in tickers:
            row = {
                'date': date,
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
    output_path = 'mock_data_temporal.parquet'
    df.to_parquet(output_path, index=False)
    print(f"Mock temporal data created: {df.shape} -> {output_path}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    return output_path

def test_dataset_splits():
    """Test the new Dataset class with temporal splits"""
    print("\n=== Testing Dataset Splits ===")
    try:
        from environments.dataset import Dataset, create_split_datasets
        
        data_path = create_mock_data()
        
        # Test individual split creation
        train_dataset = Dataset(data_path, split='train', train_end='2015-12-31', val_end='2017-12-31')
        val_dataset = Dataset(data_path, split='val', train_end='2015-12-31', val_end='2017-12-31')
        test_dataset = Dataset(data_path, split='test', train_end='2015-12-31', val_end='2017-12-31')
        
        print(f"✓ Train split: {len(train_dataset)} days, {train_dataset.num_assets} assets")
        print(f"✓ Val split: {len(val_dataset)} days, {val_dataset.num_assets} assets")
        print(f"✓ Test split: {len(test_dataset)} days, {test_dataset.num_assets} assets")
        
        # Test split creation function
        datasets = create_split_datasets(data_path, train_end='2015-12-31', val_end='2017-12-31')
        
        print(f"✓ Split creation function works")
        
        # Test that splits don't overlap
        train_dates = set(train_dataset.dates)
        val_dates = set(val_dataset.dates)
        test_dates = set(test_dataset.dates)
        
        assert len(train_dates & val_dates) == 0, "Train and val splits overlap!"
        assert len(val_dates & test_dates) == 0, "Val and test splits overlap!"
        assert len(train_dates & test_dates) == 0, "Train and test splits overlap!"
        
        print("✓ No temporal overlap between splits")
        
        # Test window sampling
        window_size = min(10, len(train_dataset))
        train_window = train_dataset.get_window(0, window_size)
        print(f"✓ Train window: features {train_window['features'].shape}, prices {train_window['raw_prices'].shape}")
        
        return datasets
        
    except Exception as e:
        print(f"Dataset splits test FAILED: {e}")
        traceback.print_exc()
        return None

def test_meta_env_splits(datasets):
    """Test MetaEnv with different split datasets"""
    print("\n=== Testing MetaEnv with Splits ===")
    try:
        from environments.env import MetaEnv
        
        # Test with train dataset
        train_dataset = datasets['train']
        
        # Create dataset tensor - ensure we have enough data for MetaEnv
        seq_len = min(15, len(train_dataset) // 2)  # Use smaller seq_len
        if seq_len < 5:
            seq_len = 5
        
        # Get a larger window for the dataset tensor
        window_size = min(seq_len * 3, len(train_dataset))  # 3x seq_len for multiple tasks
        window = train_dataset.get_window(0, window_size)
        
        dataset_tensor = {
            'features': torch.tensor(window['features'], dtype=torch.float32),
            'raw_prices': torch.tensor(window['raw_prices'], dtype=torch.float32)
        }
        
        # Create MetaEnv with smaller seq_len
        env = MetaEnv(
            dataset=dataset_tensor,
            feature_columns=train_dataset.feature_cols,
            seq_len=seq_len,
            min_horizon=3,  # Smaller horizons
            max_horizon=5
        )
        
        # Test basic functionality
        task = env.sample_task()
        env.set_task(task)
        obs = env.reset()
        
        print(f"✓ MetaEnv works with split data: obs_shape={obs.shape}")
        
        # Test step
        num_assets = obs.shape[0]
        action = np.random.rand(num_assets)
        action = action / action.sum()
        
        next_obs, reward, done, info = env.step(action)
        print(f"✓ Environment step successful: reward={reward:.4f}")
        
        return env, obs.shape
        
    except Exception as e:
        print(f"MetaEnv splits test FAILED: {e}")
        traceback.print_exc()
        return None, None

def test_train_val_pipeline():
    """Test the complete train-val pipeline"""
    print("\n=== Testing Train-Val Pipeline ===")
    try:
        from environments.dataset import create_split_datasets
        from environments.env import MetaEnv
        from models.policy import PortfolioPolicy
        from models.vae import VAE
        
        # Create mock data and splits
        data_path = create_mock_data()
        datasets = create_split_datasets(data_path, train_end='2015-12-31', val_end='2017-12-31')
        
        # Create environments for train and val
        train_dataset = datasets['train']
        val_dataset = datasets['val']
        
        seq_len = 10  # Smaller fixed seq_len
        
        # Train environment
        train_window = train_dataset.get_window(0, min(seq_len, len(train_dataset)))
        train_tensor = {
            'features': torch.tensor(train_window['features'], dtype=torch.float32),
            'raw_prices': torch.tensor(train_window['raw_prices'], dtype=torch.float32)
        }
        
        train_env = MetaEnv(
            dataset=train_tensor,
            feature_columns=train_dataset.feature_cols,
            seq_len=seq_len,
            min_horizon=5,
            max_horizon=10
        )
        
        # Val environment
        val_window = val_dataset.get_window(0, min(seq_len, len(val_dataset)))
        val_tensor = {
            'features': torch.tensor(val_window['features'], dtype=torch.float32),
            'raw_prices': torch.tensor(val_window['raw_prices'], dtype=torch.float32)
        }
        
        val_env = MetaEnv(
            dataset=val_tensor,
            feature_columns=val_dataset.feature_cols,
            seq_len=seq_len,
            min_horizon=5,
            max_horizon=10
        )
        
        # Test both environments
        print(f"Debug NEW: train_window shape: {train_tensor['features'].shape}")
        print(f"Debug NEW: val_window shape: {val_tensor['features'].shape}")
        print(f"Debug NEW: seq_len: {seq_len}")
        
        train_task = train_env.sample_task()
        train_env.set_task(train_task)
        train_obs = train_env.reset()
        
        val_task = val_env.sample_task()
        val_env.set_task(val_task)
        val_obs = val_env.reset()
        
        print(f"✓ Train env obs: {train_obs.shape}")
        print(f"✓ Val env obs: {val_obs.shape}")
        
        # Test models
        obs_shape = train_obs.shape
        latent_dim = 16
        num_assets = obs_shape[0]
        
        vae = VAE(obs_dim=obs_shape, num_assets=num_assets, latent_dim=latent_dim, hidden_dim=64)
        policy = PortfolioPolicy(obs_shape=obs_shape, latent_dim=latent_dim, num_assets=num_assets, hidden_dim=64)
        
        # Test forward pass on both datasets
        obs_tensor = torch.tensor(train_obs, dtype=torch.float32).unsqueeze(0)
        latent = torch.zeros(1, latent_dim)
        
        with torch.no_grad():
            action, value = policy.act(obs_tensor, latent, deterministic=True)
            print(f"✓ Policy works: action_sum={action.sum():.4f}")
        
        # Test VAE with mock sequence
        seq_len_test = 5
        mock_obs_seq = torch.randn(1, seq_len_test, *obs_shape)
        mock_action_seq = torch.rand(1, seq_len_test, num_assets)
        mock_action_seq = mock_action_seq / mock_action_seq.sum(dim=-1, keepdim=True)
        mock_reward_seq = torch.randn(1, seq_len_test, 1)
        
        mu, logvar, _ = vae.encode(mock_obs_seq, mock_action_seq, mock_reward_seq)
        print(f"✓ VAE encoding works: mu_shape={mu.shape}")
        
        print("✓ Complete train-val pipeline test successful")
        return True
        
    except Exception as e:
        print(f"Train-val pipeline test FAILED: {e}")
        traceback.print_exc()
        return False

def test_evaluation_function():
    """Test the evaluation function with different splits"""
    print("\n=== Testing Evaluation Function ===")
    try:
        from environments.dataset import create_split_datasets
        from environments.env import MetaEnv
        from models.policy import PortfolioPolicy
        from models.vae import VAE
        
        # Mock evaluation function (simplified version of the one in main.py)
        def mock_evaluate_on_split(env, policy, vae, num_episodes=3):
            device = torch.device("cpu")
            episode_rewards = []
            
            policy.eval()
            vae.eval()
            
            with torch.no_grad():
                for episode in range(num_episodes):
                    task = env.sample_task()
                    env.set_task(task)
                    obs = env.reset()
                    
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    latent = torch.zeros(1, 16, device=device)  # Simple zero latent
                    
                    episode_reward = 0
                    done = False
                    steps = 0
                    
                    while not done and steps < 5:  # Limit steps for testing
                        action, value = policy.act(obs_tensor, latent, deterministic=True)
                        action_cpu = action.squeeze(0).cpu().numpy()
                        
                        next_obs, reward, done, info = env.step(action_cpu)
                        episode_reward += reward
                        
                        if not done:
                            obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                        
                        steps += 1
                    
                    episode_rewards.append(episode_reward)
            
            policy.train()
            vae.train()
            
            return {
                'avg_reward': np.mean(episode_rewards),
                'rewards': episode_rewards
            }
        
        # Create test setup
        data_path = create_mock_data()
        datasets = create_split_datasets(data_path, train_end='2015-12-31', val_end='2017-12-31')
        
        # Create val environment
        val_dataset = datasets['val']
        seq_len = 8  # Small seq_len
        
        # Use larger window for dataset tensor
        val_window_size = min(seq_len * 6, len(val_dataset))  # 6x seq_len for more buffer
        val_window = val_dataset.get_window(0, val_window_size)
        val_tensor = {
            'features': torch.tensor(val_window['features'], dtype=torch.float32),
            'raw_prices': torch.tensor(val_window['raw_prices'], dtype=torch.float32)
        }
        
        val_env = MetaEnv(
            dataset=val_tensor,
            feature_columns=val_dataset.feature_cols,
            seq_len=seq_len,
            min_horizon=3,
            max_horizon=5
        )
        
        # Create models
        task = val_env.sample_task()
        val_env.set_task(task)
        obs = val_env.reset()
        
        obs_shape = obs.shape
        num_assets = obs_shape[0]
        
        vae = VAE(obs_dim=obs_shape, num_assets=num_assets, latent_dim=16, hidden_dim=32)
        policy = PortfolioPolicy(obs_shape=obs_shape, latent_dim=16, num_assets=num_assets, hidden_dim=32)
        
        # Test evaluation
        results = mock_evaluate_on_split(val_env, policy, vae, num_episodes=3)
        
        print(f"✓ Evaluation successful: avg_reward={results['avg_reward']:.4f}")
        print(f"✓ Episode rewards: {[f'{r:.4f}' for r in results['rewards']]}")
        
        return True
        
    except Exception as e:
        print(f"Evaluation function test FAILED: {e}")
        traceback.print_exc()
        return False

def test_split_data_properties():
    """Test properties specific to split data"""
    print("\n=== Testing Split Data Properties ===")
    try:
        from environments.dataset import create_split_datasets
        
        data_path = create_mock_data()
        datasets = create_split_datasets(data_path, train_end='2015-12-31', val_end='2017-12-31')
        
        # Test temporal ordering
        train_max_date = max(datasets['train'].dates)
        val_min_date = min(datasets['val'].dates)
        val_max_date = max(datasets['val'].dates)
        test_min_date = min(datasets['test'].dates)
        
        print(f"Train max date: {train_max_date.date()}")
        print(f"Val date range: {val_min_date.date()} to {val_max_date.date()}")
        print(f"Test min date: {test_min_date.date()}")
        
        # Verify temporal ordering
        assert train_max_date < val_min_date, "Train-val temporal ordering violated"
        assert val_max_date < test_min_date, "Val-test temporal ordering violated"
        print("✓ Temporal ordering correct")
        
        # Test consistent asset counts
        train_assets = datasets['train'].num_assets
        val_assets = datasets['val'].num_assets
        test_assets = datasets['test'].num_assets
        
        assert train_assets == val_assets == test_assets, "Inconsistent asset counts across splits"
        print(f"✓ Consistent asset count: {train_assets}")
        
        # Test feature consistency
        train_features = set(datasets['train'].feature_cols)
        val_features = set(datasets['val'].feature_cols)
        test_features = set(datasets['test'].feature_cols)
        
        assert train_features == val_features == test_features, "Inconsistent features across splits"
        print(f"✓ Consistent features: {len(train_features)} features")
        
        return True
        
    except Exception as e:
        print(f"Split data properties test FAILED: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all split-related tests"""
    print("=== VariBAD Portfolio Debug Test - Train-Test-Val Splits ===")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Force CPU usage
    if torch.cuda.is_available():
        print("CUDA detected but forcing CPU usage for debugging")
        torch.cuda.is_available = lambda: False
    
    try:
        # Test dataset splits
        datasets = test_dataset_splits()
        if datasets is None:
            print("Cannot proceed - Dataset splits failed")
            return
        
        # Test MetaEnv with splits
        env, obs_shape = test_meta_env_splits(datasets)
        if env is None:
            print("Cannot proceed - MetaEnv splits failed")
            return
        
        # Test split-specific functionality
        success = True
        success &= test_split_data_properties()
        success &= test_train_val_pipeline()
        success &= test_evaluation_function()
        
        if success:
            print("\n✅ All split-related tests passed!")
            print("✅ Ready for training with train-test-val splits!")
        else:
            print("\n❌ Some split tests failed")
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
    
    finally:
        # Cleanup
        for mock_file in ['mock_data_temporal.parquet']:
            mock_path = Path(mock_file)
            if mock_path.exists():
                mock_path.unlink()
                print(f"Cleaned up {mock_file}")

if __name__ == "__main__":
    main()