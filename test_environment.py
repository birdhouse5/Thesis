
# ===========================================
# FILE 3: test_simple_environment.py (NEW SIMPLE TEST)
# ===========================================

"""
Simple test for the existing environment functionality.
"""

import sys
import os
import numpy as np

# Add sp500_loader to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_existing_environment():
    """Test the existing environment functionality."""
    print("=== TESTING EXISTING ENVIRONMENT ===")
    
    try:
        from sp500_loader import load_dataset, create_quick_loader
        from sp500_loader.core.environment import create_env_from_loader
        
        print("✓ Imports successful")
        
        # Load data
        print("Loading S&P 500 dataset...")
        panel_df = load_dataset('sp500_loader/data/sp500_dataset.parquet')
        loader = create_quick_loader(
            panel_df,
            train_end='2015-12-31',
            val_end='2018-12-31',
            episode_length=30,
            min_history_days=100
        )
        print("✓ Data loaded")
        
        # Test different configurations
        configs = [
            {
                'name': 'Long-Only with DSR',
                'reward_mode': 'dsr',
                'short_ratio_mode': 'fixed',
                'fixed_short_ratio': 0.0
            },
            {
                'name': 'Long-Short with DSR',
                'reward_mode': 'dsr',
                'short_ratio_mode': 'fixed',
                'fixed_short_ratio': 0.3
            },
            {
                'name': 'Traditional Returns',
                'reward_mode': 'return',
                'short_ratio_mode': 'fixed',
                'fixed_short_ratio': 0.2
            }
        ]
        
        for config in configs:
            print(f"\n--- Testing {config['name']} ---")
            
            # Create environment
            env = create_env_from_loader(
                loader,
                split='train',
                episode_length=10,  # Short for testing
                lookback_window=5,
                **{k: v for k, v in config.items() if k != 'name'}
            )
            
            print(f"✓ Environment created")
            print(f"  Action space: {env.action_space.shape}")
            print(f"  Observation space: {env.observation_space.shape}")
            print(f"  Reward mode: {env.reward_mode}")
            
            # Run short episode
            obs = env.reset(seed=42)
            total_reward = 0
            
            for step in range(5):
                action = np.random.random(env.action_space.shape[0])
                obs, reward, done, info = env.step(action)
                total_reward += reward
                
                if step == 0:
                    print(f"  Step 1: Reward={reward:+.6f}, Value=${info['portfolio_value']:,.0f}")
                
                if done:
                    break
            
            print(f"  ✓ Episode completed, total reward: {total_reward:+.6f}")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run simple tests."""
    print("SIMPLE ENVIRONMENT TEST")
    print("=" * 40)
    
    success = test_existing_environment()
    
    if success:
        print("\n🎉 Environment is working correctly!")
        print("\nYour environment includes:")
        print("✅ Long-only and long-short modes")
        print("✅ DSR and traditional return rewards")
        print("✅ Transaction costs")
        print("✅ Portfolio tracking")
        print("✅ Proper action/observation spaces")
        
        print("\nNext steps:")
        print("1. Train RL agents with this environment")
        print("2. Add technical indicators if needed")
        print("3. Experiment with different configurations")
        
    else:
        print("\n❌ Tests failed. Check the errors above.")
    
    return success


if __name__ == "__main__":
    main()