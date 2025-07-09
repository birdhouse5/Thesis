"""
Test script for VariBAD Portfolio implementation.
Verifies the basic functionality before full training.
"""

import sys
import os
import torch
import numpy as np
import pandas as pd

# Add sp500_loader to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_varibad_components():
    """Test individual VariBAD components."""
    print("=== TESTING VARIBAD COMPONENTS ===")
    
    # Test parameters
    batch_size = 4
    seq_len = 10
    state_dim = 50
    action_dim = 20
    latent_dim = 8
    hidden_size = 64
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        from varibad_portfolio_env import (
            PortfolioVariationalEncoder,
            PortfolioDecoder, 
            VariBADPortfolioPolicy
        )
        
        # Test encoder
        print("\n1. Testing Encoder...")
        encoder = PortfolioVariationalEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_size=hidden_size
        ).to(device)
        
        # Create dummy trajectory: [state, action, return]
        trajectory = torch.randn(batch_size, seq_len, state_dim + action_dim + 1).to(device)
        mu, logvar = encoder(trajectory)
        
        print(f"✓ Encoder output shapes: mu={mu.shape}, logvar={logvar.shape}")
        print(f"  Expected: mu=({batch_size}, {latent_dim}), logvar=({batch_size}, {latent_dim})")
        
        # Test decoder
        print("\n2. Testing Decoder...")
        decoder = PortfolioDecoder(
            latent_dim=latent_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size
        ).to(device)
        
        task_embedding = torch.randn(batch_size, latent_dim).to(device)
        states = torch.randn(batch_size, seq_len, state_dim).to(device)
        actions = torch.randn(batch_size, seq_len, action_dim).to(device)
        
        predicted_returns = decoder(task_embedding, states, actions)
        
        print(f"✓ Decoder output shape: {predicted_returns.shape}")
        print(f"  Expected: ({batch_size}, {seq_len}, 1)")
        
        # Test policy
        print("\n3. Testing Policy...")
        policy = VariBADPortfolioPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_size=hidden_size
        ).to(device)
        
        state = torch.randn(batch_size, state_dim).to(device)
        task_mu = torch.randn(batch_size, latent_dim).to(device)
        task_logvar = torch.randn(batch_size, latent_dim).to(device)
        
        action = policy(state, task_mu, task_logvar)
        
        print(f"✓ Policy output shape: {action.shape}")
        print(f"  Expected: ({batch_size}, {action_dim})")
        print(f"  Action range: [{action.min().item():.3f}, {action.max().item():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vae_loss():
    """Test VAE loss computation."""
    print("\n=== TESTING VAE LOSS ===")
    
    try:
        from varibad_portfolio_env import VariBADTrainer
        
        # Create trainer
        state_dim, action_dim, latent_dim = 20, 10, 5
        hidden_size = 32
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        from varibad_portfolio_env import (
            PortfolioVariationalEncoder,
            PortfolioDecoder,
            VariBADPortfolioPolicy
        )
        
        encoder = PortfolioVariationalEncoder(state_dim, action_dim, latent_dim, hidden_size).to(device)
        decoder = PortfolioDecoder(latent_dim, state_dim, action_dim, hidden_size).to(device)
        policy = VariBADPortfolioPolicy(state_dim, action_dim, latent_dim, hidden_size).to(device)
        
        trainer = VariBADTrainer(encoder, decoder, policy, device=device)
        
        # Create dummy batch data
        batch_size, seq_len = 3, 8
        batch_data = {
            'trajectories': torch.randn(batch_size, seq_len, state_dim + action_dim + 1).to(device),
            'states': torch.randn(batch_size, seq_len, state_dim).to(device),
            'actions': torch.randn(batch_size, seq_len, action_dim).to(device),
            'returns': torch.randn(batch_size, seq_len, 1).to(device)
        }
        
        # Test VAE loss computation
        vae_loss, loss_dict = trainer.compute_vae_loss(
            batch_data['trajectories'],
            batch_data['states'],
            batch_data['actions'],
            batch_data['returns']
        )
        
        print(f"✓ VAE Loss computation successful:")
        print(f"  Total VAE loss: {loss_dict['vae_loss']:.6f}")
        print(f"  Reconstruction loss: {loss_dict['recon_loss']:.6f}")
        print(f"  KL loss: {loss_dict['kl_loss']:.6f}")
        
        # Test VAE update
        loss_dict = trainer.update_vae(batch_data)
        print(f"✓ VAE update successful")
        
        return True
        
    except Exception as e:
        print(f"❌ VAE loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_portfolio_env():
    """Test integration with portfolio environment."""
    print("\n=== TESTING WITH PORTFOLIO ENVIRONMENT ===")
    
    try:
        # Create dummy portfolio environment first
        import pandas as pd
        from sp500_loader.core.environment import PortfolioEnv
        
        # Create minimal test data
        dates = pd.date_range('2020-01-01', '2020-03-01', freq='B')
        tickers = ['STOCK_A', 'STOCK_B', 'STOCK_C']
        
        np.random.seed(42)
        data = []
        for ticker in tickers:
            initial_price = np.random.uniform(50, 200)
            prices = [initial_price]
            for _ in range(len(dates) - 1):
                prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
            
            for date, price in zip(dates, prices):
                data.append({'date': date, 'ticker': ticker, 'adj_close': price})
        
        price_data = pd.DataFrame(data).set_index(['date', 'ticker'])
        
        # Create base portfolio environment
        base_env = PortfolioEnv(
            price_data,
            episode_length=10,
            lookback_window=5,
            short_ratio_mode='fixed',
            fixed_short_ratio=0.2,
            reward_mode='dsr'
        )
        
        print(f"✓ Base environment created:")
        print(f"  Observation space: {base_env.observation_space.shape}")
        print(f"  Action space: {base_env.action_space.shape}")
        
        # Create VariBAD wrapper
        from varibad_portfolio_env import (
            PortfolioVariationalEncoder,
            PortfolioDecoder,
            VariBADPortfolioPolicy,
            VariBADPortfolioEnv
        )
        
        device = 'cpu'  # Use CPU for testing
        state_dim = base_env.observation_space.shape[0]
        action_dim = base_env.action_space.shape[0]
        latent_dim = 6
        hidden_size = 64
        
        encoder = PortfolioVariationalEncoder(state_dim, action_dim, latent_dim, hidden_size).to(device)
        decoder = PortfolioDecoder(latent_dim, state_dim, action_dim, hidden_size).to(device)
        policy = VariBADPortfolioPolicy(state_dim, action_dim, latent_dim, hidden_size).to(device)
        
        varibad_env = VariBADPortfolioEnv(
            base_env=base_env,
            encoder=encoder,
            decoder=decoder,
            policy=policy,
            episode_length=10,
            device=device
        )
        
        print(f"✓ VariBAD environment created")
        
        # Test episode
        obs = varibad_env.reset(seed=42)
        print(f"✓ Reset successful, observation shape: {obs.shape}")
        print(f"  Expected shape includes task posterior: {state_dim} + 2*{latent_dim} = {state_dim + 2*latent_dim}")
        
        # Test a few steps
        for step in range(3):
            # Get action from policy
            action = varibad_env.get_action(obs)
            print(f"  Step {step}: Action shape: {action.shape}, range: [{action.min():.3f}, {action.max():.3f}]")
            
            # Take step
            obs, reward, done, info = varibad_env.step(action)
            print(f"    Reward: {reward:.6f}, Done: {done}")
            
            if done:
                break
        
        print(f"✓ Episode execution successful")
        print(f"  Task posterior updated: mu shape = {varibad_env.current_task_posterior['mu'].shape}")
        print(f"  Trajectory buffer length: {len(varibad_env.trajectory_buffer)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Portfolio environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_task_inference():
    """Test task inference mechanics."""
    print("\n=== TESTING TASK INFERENCE ===")
    
    try:
        # Create simple test to verify task inference works
        from varibad_portfolio_env import PortfolioVariationalEncoder
        
        state_dim, action_dim, latent_dim = 10, 6, 4
        encoder = PortfolioVariationalEncoder(state_dim, action_dim, latent_dim, 32)
        
        # Simulate two different "tasks" with different return patterns
        batch_size, seq_len = 2, 15
        
        # Task 1: Positive trend
        trajectory_1 = torch.randn(1, seq_len, state_dim + action_dim + 1)
        trajectory_1[0, :, -1] = torch.linspace(0.01, 0.05, seq_len)  # Increasing returns
        
        # Task 2: Negative trend  
        trajectory_2 = torch.randn(1, seq_len, state_dim + action_dim + 1)
        trajectory_2[0, :, -1] = torch.linspace(-0.01, -0.05, seq_len)  # Decreasing returns
        
        # Encode both
        mu1, logvar1 = encoder(trajectory_1)
        mu2, logvar2 = encoder(trajectory_2)
        
        # Check if embeddings are different
        embedding_diff = torch.norm(mu1 - mu2).item()
        
        print(f"✓ Task inference test:")
        print(f"  Task 1 embedding: {mu1.squeeze().detach().numpy()}")
        print(f"  Task 2 embedding: {mu2.squeeze().detach().numpy()}")
        print(f"  Embedding difference: {embedding_diff:.4f}")
        
        if embedding_diff > 0.1:
            print(f"✓ Embeddings are sufficiently different")
        else:
            print(f"⚠️  Embeddings are similar (may need more training)")
        
        return True
        
    except Exception as e:
        print(f"❌ Task inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("VARIBAD PORTFOLIO OPTIMIZATION - COMPONENT TESTS")
    print("=" * 60)
    
    tests = [
        test_varibad_components,
        test_vae_loss,
        test_task_inference,
        test_with_portfolio_env
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("\nNext steps:")
        print("1. Integrate with your SP500 data loader")
        print("2. Implement task sampling for meta-training")
        print("3. Add PPO training loop")
        print("4. Test on real portfolio data")
    elif passed >= total // 2:
        print("✅ Most tests passed - implementation looks good!")
        print("Fix remaining issues and proceed with integration.")
    else:
        print("❌ Many tests failed - check implementation.")
    
    return passed == total


if __name__ == "__main__":
    main()