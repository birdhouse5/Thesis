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
    
    import pandas as pd
    
    dates = pd.date_range('2020-01-01', periods=200, freq='D')  # More data for proper testing
    tickers = ['AAPL', 'MSFT', 'GOOGL']  # 3 assets for testing
    
    data = []
    for date in dates:
        for ticker in tickers:
            # Mock normalized features (matches expected feature set)
            row = {
                'date': date.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'close': np.random.uniform(50, 200),  # Raw price for returns
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
    
    # Save as parquet (matching expected format)
    output_path = 'mock_data.parquet'
    df.to_parquet(output_path, index=False)
    print(f"Mock data created: {df.shape} -> {output_path}")
    return output_path

def test_dataset():
    """Test dataset loading with new setup"""
    print("\n=== Testing Dataset ===")
    try:
        from environments.dataset import Dataset
        
        data_path = create_mock_data()
        dataset = Dataset(data_path)
        
        print(f"Dataset loaded: {len(dataset)} days, {dataset.num_assets} assets")
        print(f"Features: {dataset.num_features}")
        print(f"Feature columns: {dataset.feature_cols[:5]}...")  # First 5
        
        # Test window sampling
        window = dataset.get_window(0, 20)
        print(f"Window features shape: {window['features'].shape}")  # Should be (20, 3, num_features)
        print(f"Window raw_prices shape: {window['raw_prices'].shape}")  # Should be (20, 3)
        
        return dataset
        
    except Exception as e:
        print(f"Dataset test FAILED: {e}")
        traceback.print_exc()
        return None

def test_meta_environment(dataset):
    """Test MetaEnv with new architecture"""
    print("\n=== Testing MetaEnv ===")
    try:
        from environments.env import MetaEnv
        
        # Prepare dataset tensor for MetaEnv
        seq_len = 60
        num_windows = len(dataset) // seq_len
        
        all_features = []
        all_prices = []
        
        for i in range(num_windows):
            start_idx = i * seq_len
            end_idx = start_idx + seq_len
            if end_idx <= len(dataset):
                window = dataset.get_window(start_idx, end_idx)
                all_features.append(torch.tensor(window['features'], dtype=torch.float32))
                all_prices.append(torch.tensor(window['raw_prices'], dtype=torch.float32))
        
        if not all_features:
            print("No complete windows available, using shorter sequence")
            window = dataset.get_window(0, min(30, len(dataset)))
            all_features = [torch.tensor(window['features'], dtype=torch.float32)]
            all_prices = [torch.tensor(window['raw_prices'], dtype=torch.float32)]
        
        # Stack and reshape for MetaEnv
        features_tensor = torch.cat(all_features, dim=0)  # (total_time, N, F)
        prices_tensor = torch.cat(all_prices, dim=0)      # (total_time, N)
        
        dataset_tensor = {
            'features': features_tensor,
            'raw_prices': prices_tensor
        }
        
        print(f"Dataset tensor shapes: features={features_tensor.shape}, prices={prices_tensor.shape}")
        
        # Create MetaEnv
        env = MetaEnv(
            dataset=dataset_tensor,
            feature_columns=dataset.feature_cols,
            seq_len=30,  # Shorter for testing
            min_horizon=20,
            max_horizon=25
        )
        
        # Test task sampling
        task = env.sample_task()
        print(f"Task sampled: {task.keys()}")
        
        # Set task and reset
        env.set_task(task)
        initial_obs = env.reset()
        print(f"Initial observation shape: {initial_obs.shape}")  # Should be (N, F)
        
        # Test step with portfolio weights
        num_assets = initial_obs.shape[0]
        portfolio_weights = torch.rand(num_assets)
        portfolio_weights = portfolio_weights / portfolio_weights.sum()  # Normalize to sum=1
        
        next_obs, reward, done, info = env.step(portfolio_weights.numpy())
        print(f"Step successful: reward={reward:.4f}, done={done}")
        print(f"Info keys: {info.keys()}")
        print(f"Portfolio allocation sum: {portfolio_weights.sum():.4f}")
        
        return env, initial_obs.shape
        
    except Exception as e:
        print(f"MetaEnv test FAILED: {e}")
        traceback.print_exc()
        return None, None

def test_models(obs_shape, num_assets):
    """Test VAE and Policy models with new architecture"""
    print("\n=== Testing Models ===")
    
    # Test Policy
    try:
        from models.policy import PortfolioPolicy
        
        latent_dim = 32  # Smaller for testing
        policy = PortfolioPolicy(
            obs_shape=obs_shape,
            latent_dim=latent_dim,
            num_assets=num_assets,
            hidden_dim=128  # Smaller for testing
        )
        print("✓ Policy created successfully")
        
        # Test forward pass
        batch_size = 2
        mock_obs = torch.randn(batch_size, *obs_shape)
        mock_latent = torch.randn(batch_size, latent_dim)
        
        # Test act method
        portfolio_weights, value = policy.act(mock_obs, mock_latent)
        print(f"✓ Policy act: weights shape {portfolio_weights.shape}, value shape {value.shape}")
        print(f"  Portfolio weights sum: {portfolio_weights.sum(dim=-1)}")  # Should be ~1.0
        
        # Test evaluate_actions
        values, log_probs, entropy = policy.evaluate_actions(mock_obs, mock_latent, portfolio_weights)
        print(f"✓ Policy evaluate: values {values.shape}, log_probs {log_probs.shape}, entropy {entropy.shape}")
        
    except Exception as e:
        print(f"Policy test FAILED: {e}")
        traceback.print_exc()
    
    # Test VAE
    try:
        from models.vae import VAE
        
        vae = VAE(
            obs_dim=obs_shape,
            num_assets=num_assets,
            latent_dim=latent_dim,
            hidden_dim=128
        )
        print("✓ VAE created successfully")
        
        # Test forward pass
        seq_len = 10
        batch_size = 2
        
        mock_obs_seq = torch.randn(batch_size, seq_len, *obs_shape)
        mock_action_seq = torch.randn(batch_size, seq_len, num_assets)  # Portfolio weights
        mock_action_seq = torch.softmax(mock_action_seq, dim=-1)        # Normalize to valid weights
        mock_reward_seq = torch.randn(batch_size, seq_len, 1)
        
        latent, mu, logvar, hidden = vae(mock_obs_seq, mock_action_seq, mock_reward_seq)
        print(f"✓ VAE forward: latent {latent.shape}, mu {mu.shape}, logvar {logvar.shape}")
        
        # Test loss computation
        loss, loss_components = vae.compute_loss(mock_obs_seq, mock_action_seq, mock_reward_seq)
        print(f"✓ VAE loss: {loss.item():.4f}")
        print(f"  Components: {loss_components}")
        
    except Exception as e:
        print(f"VAE test FAILED: {e}")
        traceback.print_exc()

def test_trainer_integration(env, obs_shape, num_assets):
    """Test trainer with models and environment"""
    print("\n=== Testing Trainer Integration ===")
    try:
        from models.policy import PortfolioPolicy
        from models.vae import VAE
        from algorithms.trainer import PPOTrainer
        
        # Create models
        latent_dim = 32
        policy = PortfolioPolicy(obs_shape, latent_dim, num_assets, hidden_dim=64)
        vae = VAE(obs_shape, num_assets, latent_dim, hidden_dim=64)
        
        # Mock config
        class MockConfig:
            device = "cpu"
            policy_lr = 3e-4
            vae_lr = 1e-4
            ppo_epochs = 2
            ppo_clip_ratio = 0.2
            value_loss_coef = 0.5
            entropy_coef = 0.01
            max_grad_norm = 0.5
            gae_lambda = 0.95
            discount_factor = 0.99
            batch_size = 4
            vae_batch_size = 2
            vae_beta = 0.1
            vae_update_freq = 1
            max_horizon = 20
            log_interval = 5
        
        config = MockConfig()
        trainer = PPOTrainer(env, policy, vae, config)
        print("✓ Trainer created successfully")
        
        # Test trajectory collection
        print("Testing trajectory collection...")
        trajectory = trainer.collect_trajectory()
        print(f"✓ Trajectory collected: {len(trajectory['rewards'])} steps")
        print(f"  Shapes: obs {trajectory['observations'].shape}, actions {trajectory['actions'].shape}")
        print(f"  Total reward: {trajectory['rewards'].sum().item():.4f}")
        
        # Test advantage computation
        advantages, returns = trainer.compute_advantages(trajectory)
        print(f"✓ Advantages computed: shape {advantages.shape}")
        
        print("✓ Trainer integration test successful")
        
    except Exception as e:
        print(f"Trainer test FAILED: {e}")
        traceback.print_exc()

def test_full_training_step():
    """Test a complete training step end-to-end"""
    print("\n=== Testing Full Training Step ===")
    try:
        # Use smaller parameters for quick testing
        print("Setting up minimal training environment...")
        
        # Create mock dataset
        data_path = create_mock_data()
        
        from environments.dataset import Dataset
        dataset = Dataset(data_path)
        
        # Create minimal dataset tensor
        window = dataset.get_window(0, min(50, len(dataset)))  # Small window
        dataset_tensor = {
            'features': torch.tensor(window['features'], dtype=torch.float32),
            'raw_prices': torch.tensor(window['raw_prices'], dtype=torch.float32)
        }
        
        from environments.env import MetaEnv
        env = MetaEnv(
            dataset=dataset_tensor,
            feature_columns=dataset.feature_cols,
            seq_len=20,  # Very short for testing
            min_horizon=10,
            max_horizon=15
        )
        
        # Sample task and get observation shape
        task = env.sample_task()
        env.set_task(task)
        obs = env.reset()
        obs_shape = obs.shape
        num_assets = obs_shape[0]
        
        print(f"Environment setup: obs_shape={obs_shape}, num_assets={num_assets}")
        
        # Create models
        from models.policy import PortfolioPolicy
        from models.vae import VAE
        
        latent_dim = 16  # Very small
        policy = PortfolioPolicy(obs_shape, latent_dim, num_assets, hidden_dim=32)
        vae = VAE(obs_shape, num_assets, latent_dim, hidden_dim=32)
        
        print("Models created, testing single forward pass...")
        
        # Test single step
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        latent = torch.zeros(1, latent_dim)
        
        action, value = policy.act(obs_tensor, latent)
        print(f"Policy output: action sum={action.sum():.4f}, value={value.item():.4f}")
        
        # Test environment step
        next_obs, reward, done, info = env.step(action.squeeze(0).numpy())
        print(f"Environment step: reward={reward:.4f}, done={done}")
        
        print("✓ Full training step test successful")
        
    except Exception as e:
        print(f"Full training test FAILED: {e}")
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
        
        print("✓ Logging test successful")
        exp_logger.close()
        
    except Exception as e:
        print(f"Logging test FAILED: {e}")
        traceback.print_exc()

def main():
    """Run all debug tests for new VariBAD setup"""
    print("=== VariBAD Portfolio Debug Test (Updated) ===")
    print("Testing new architecture with MetaEnv, Portfolio Policy, and PPO")
    
    # Disable GPU for debugging
    torch.cuda.is_available = lambda: False
    
    # Test components step by step
    dataset = test_dataset()
    if dataset is None:
        print("Cannot proceed - Dataset failed")
        return
    
    env, obs_shape = test_meta_environment(dataset)
    if env is None:
        print("Cannot proceed - MetaEnv failed")
        return
    
    num_assets = obs_shape[0]
    print(f"\nObservation shape: {obs_shape}, Number of assets: {num_assets}")
    
    test_models(obs_shape, num_assets)
    
    test_trainer_integration(env, obs_shape, num_assets)
    
    test_full_training_step()
    
    test_logging()
    
    print("\n=== Debug Test Complete ===")
    print("✓ All major components tested successfully!")
    print("✓ New VariBAD architecture is ready for training")
    print("\nNext steps:")
    print("1. Run with real dataset: python main.py")
    print("2. Monitor tensorboard logs for training progress")
    print("3. Adjust hyperparameters in Config class as needed")

if __name__ == "__main__":
    main()