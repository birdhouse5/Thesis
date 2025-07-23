"""
Debug script to identify and fix the dimension mismatch in VariBAD training.

This script will help you understand exactly where the dimension mismatch is occurring
and provide the correct dimensions to use in your models.
"""

import pandas as pd
import numpy as np
import torch
from test_MPD_setup import MetaTraderPortfolioMDP


def debug_mdp_dimensions():
    """Debug the MDP to understand actual state and action dimensions."""
    print("🔍 DEBUGGING MDP DIMENSIONS")
    print("=" * 50)
    
    # Load your data
    try:
        data = pd.read_parquet('data/sp500_rl_ready_cleaned.parquet')
        print(f"✓ Data loaded: {data.shape}")
    except FileNotFoundError:
        print("❌ Data file not found. Please ensure sp500_rl_ready_cleaned.parquet exists.")
        return None
    
    # Create MDP with different configurations
    print(f"\n1. Testing Long-Only MDP...")
    
    env_long = MetaTraderPortfolioMDP(
        data=data,
        episode_length=15,
        short_selling_enabled=False
    )
    
    state_long = env_long.reset()
    action_long = env_long.action_space.sample()
    
    print(f"   Long-only state dimension: {len(state_long)}")
    print(f"   Long-only action dimension: {len(action_long)}")
    print(f"   Action space: {env_long.action_space}")
    
    print(f"\n2. Testing Long/Short MDP...")
    
    env_short = MetaTraderPortfolioMDP(
        data=data,
        episode_length=15,
        short_selling_enabled=True,
        max_short_ratio=0.3
    )
    
    state_short = env_short.reset()
    action_short = env_short.action_space.sample()
    
    print(f"   Long/short state dimension: {len(state_short)}")
    print(f"   Long/short action dimension: {len(action_short)}")
    print(f"   Action space: {env_short.action_space}")
    
    print(f"\n3. Analyzing State Components...")
    
    # Get the detailed breakdown from your MDP
    print(f"   Asset features: {len(env_short.asset_feature_cols)} × {env_short.n_assets} = {len(env_short.asset_feature_cols) * env_short.n_assets}")
    print(f"   Market features: {len(env_short.market_feature_cols)}")
    print(f"   Account features: {len(action_short)} (previous portfolio weights)")
    
    total_expected = (len(env_short.asset_feature_cols) * env_short.n_assets + 
                     len(env_short.market_feature_cols) + 
                     len(action_short))
    
    print(f"   Expected total: {total_expected}")
    print(f"   Actual total: {len(state_short)}")
    
    if total_expected != len(state_short):
        print(f"   ⚠️ Dimension mismatch in state construction!")
    else:
        print(f"   ✓ State dimensions match expected calculation")
    
    print(f"\n4. Encoder Input Dimension Calculation...")
    
    # This is what goes into the encoder
    state_dim = len(state_short)
    action_dim = len(action_short)
    reward_dim = 1
    
    encoder_input_dim = state_dim + action_dim + reward_dim
    
    print(f"   State: {state_dim}")
    print(f"   Action: {action_dim}")
    print(f"   Reward: {reward_dim}")
    print(f"   Total encoder input: {encoder_input_dim}")
    
    return {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'encoder_input_dim': encoder_input_dim,
        'env': env_short
    }


def debug_trajectory_sequence():
    """Debug trajectory sequence construction to find dimension issues."""
    print(f"\n🔍 DEBUGGING TRAJECTORY SEQUENCE")
    print("=" * 50)
    
    # Get MDP dimensions
    dims = debug_mdp_dimensions()
    if dims is None:
        return
    
    env = dims['env']
    state_dim = dims['state_dim']
    action_dim = dims['action_dim']
    
    print(f"\n1. Collecting sample episode...")
    
    # Collect a short episode
    states = []
    actions = []
    rewards = []
    
    state = env.reset()
    states.append(state.copy())
    
    for step in range(5):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        
        actions.append(action.copy())
        rewards.append(reward)
        
        print(f"   Step {step}: state {len(state)}, action {len(action)}, reward scalar")
        
        if len(state) != state_dim:
            print(f"   ❌ State dimension changed! Expected {state_dim}, got {len(state)}")
        
        if len(action) != action_dim:
            print(f"   ❌ Action dimension mismatch! Expected {action_dim}, got {len(action)}")
        
        state = next_state
        states.append(state.copy())
        
        if done:
            break
    
    print(f"\n2. Converting to tensors...")
    
    # Convert to tensors like the trainer does
    try:
        states_tensor = torch.FloatTensor(np.array(states[:-1]))  # Exclude last state
        actions_tensor = torch.FloatTensor(np.array(actions))
        rewards_tensor = torch.FloatTensor(rewards)
        
        print(f"   States tensor: {states_tensor.shape}")
        print(f"   Actions tensor: {actions_tensor.shape}")
        print(f"   Rewards tensor: {rewards_tensor.shape}")
        
        # Add batch dimension
        batch_states = states_tensor.unsqueeze(0)  # [1, seq_len, state_dim]
        batch_actions = actions_tensor.unsqueeze(0)  # [1, seq_len, action_dim]
        batch_rewards = rewards_tensor.unsqueeze(0)  # [1, seq_len]
        
        print(f"   Batched states: {batch_states.shape}")
        print(f"   Batched actions: {batch_actions.shape}")
        print(f"   Batched rewards: {batch_rewards.shape}")
        
        # Test encoder input concatenation
        rewards_expanded = batch_rewards.unsqueeze(-1)  # [1, seq_len, 1]
        concatenated = torch.cat([batch_states, batch_actions, rewards_expanded], dim=-1)
        
        print(f"   Concatenated input: {concatenated.shape}")
        print(f"   Expected: [1, {len(actions)}, {state_dim + action_dim + 1}]")
        
        actual_input_dim = concatenated.shape[-1]
        expected_input_dim = state_dim + action_dim + 1
        
        if actual_input_dim != expected_input_dim:
            print(f"   ❌ FOUND THE ISSUE!")
            print(f"   Actual concatenated dim: {actual_input_dim}")
            print(f"   Expected encoder input: {expected_input_dim}")
            print(f"   Difference: {actual_input_dim - expected_input_dim}")
        else:
            print(f"   ✓ Concatenation dimensions are correct")
            
        return {
            'actual_input_dim': actual_input_dim,
            'expected_input_dim': expected_input_dim,
            'state_dim': state_dim,
            'action_dim': action_dim
        }
        
    except Exception as e:
        print(f"   ❌ Tensor conversion failed: {e}")
        return None


def test_encoder_with_correct_dims():
    """Test encoder with the correct dimensions."""
    print(f"\n🔍 TESTING ENCODER WITH CORRECT DIMENSIONS")
    print("=" * 50)
    
    # Get the actual dimensions
    trajectory_info = debug_trajectory_sequence()
    if trajectory_info is None:
        return
    
    state_dim = trajectory_info['state_dim']
    action_dim = trajectory_info['action_dim']
    actual_input_dim = trajectory_info['actual_input_dim']
    
    print(f"\n1. Creating encoder with correct dimensions...")
    
    from varibad_models import TrajectoryEncoder
    
    # Create encoder with the ACTUAL input dimension
    try:
        encoder = TrajectoryEncoder(
            state_dim=state_dim,
            action_dim=action_dim,  # This will calculate input_dim correctly
            latent_dim=5,
            hidden_dim=128
        )
        
        print(f"   ✓ Encoder created successfully")
        print(f"   Encoder expects input_dim: {encoder.input_dim}")
        print(f"   Actual trajectory input_dim: {actual_input_dim}")
        
        if encoder.input_dim == actual_input_dim:
            print(f"   ✅ DIMENSIONS MATCH! The encoder is configured correctly.")
        else:
            print(f"   ❌ Still a mismatch!")
            print(f"   The issue might be in how action_dim is calculated.")
            
        return encoder
        
    except Exception as e:
        print(f"   ❌ Encoder creation failed: {e}")
        return None


def provide_fix_recommendations():
    """Provide specific recommendations to fix the dimension issue."""
    print(f"\n💡 FIX RECOMMENDATIONS")
    print("=" * 50)
    
    # Run the full debug
    trajectory_info = debug_trajectory_sequence()
    encoder = test_encoder_with_correct_dims()
    
    if trajectory_info is None:
        print("❌ Could not complete diagnosis. Check your data file and MDP setup.")
        return
    
    state_dim = trajectory_info['state_dim']
    action_dim = trajectory_info['action_dim']
    actual_input_dim = trajectory_info['actual_input_dim']
    expected_input_dim = trajectory_info['expected_input_dim']
    
    print(f"\n🎯 SUMMARY:")
    print(f"   State dimension: {state_dim}")
    print(f"   Action dimension: {action_dim}")
    print(f"   Actual encoder input: {actual_input_dim}")
    print(f"   Expected encoder input: {expected_input_dim}")
    
    if actual_input_dim != expected_input_dim:
        print(f"\n🔧 FIXES NEEDED:")
        print(f"   1. Update varibad_models.py TrajectoryEncoder:")
        print(f"      self.input_dim = {actual_input_dim}  # Instead of state_dim + action_dim + 1")
        print(f"   ")
        print(f"   2. Or debug why action_dim calculation is wrong:")
        print(f"      Expected action_dim for input: {actual_input_dim - state_dim - 1}")
        print(f"      Actual action_dim: {action_dim}")
        print(f"      Difference: {action_dim - (actual_input_dim - state_dim - 1)}")
    else:
        print(f"   ✅ No fixes needed - dimensions are correct!")
    
    print(f"\n📝 RECOMMENDED VARIBAD_TRAINER.PY SETTINGS:")
    print(f"   VariBADTrainer(")
    print(f"       state_dim={state_dim},")
    print(f"       action_dim={action_dim},  # Total action dimension from environment")
    print(f"       ...)")
    
    print(f"\n📝 RECOMMENDED VARIBAD_MODELS.PY SETTINGS:")
    print(f"   TrajectoryEncoder(")
    print(f"       state_dim={state_dim},")
    print(f"       action_dim={action_dim},")
    print(f"       ...)")
    print(f"   # This should give input_dim = {actual_input_dim}")


def quick_fix_test():
    """Test if the issue is simply wrong input_dim in encoder."""
    print(f"\n🚀 QUICK FIX TEST")
    print("=" * 50)
    
    # Get dimensions
    dims = debug_mdp_dimensions()
    if dims is None:
        return
    
    state_dim = dims['state_dim']
    action_dim = dims['action_dim']
    
    # The error shows expected 995 but got 1025
    # So the encoder was configured for 995 but receiving 1025
    
    expected_by_encoder = 995
    actual_from_trajectory = 1025
    difference = actual_from_trajectory - expected_by_encoder
    
    print(f"   Error analysis:")
    print(f"   Encoder expected: {expected_by_encoder}")
    print(f"   Trajectory provided: {actual_from_trajectory}")
    print(f"   Difference: {difference}")
    
    print(f"\n   Current MDP dimensions:")
    print(f"   State: {state_dim}")
    print(f"   Action: {action_dim}")
    print(f"   Expected encoder input: {state_dim + action_dim + 1}")
    
    if state_dim + action_dim + 1 == actual_from_trajectory:
        print(f"   ✅ MDP dimensions match the actual trajectory input!")
        print(f"   The fix is to update the encoder to expect {actual_from_trajectory} inputs.")
    else:
        print(f"   ❌ Still investigating...")
    
    print(f"\n🔧 IMMEDIATE FIX:")
    print(f"   In varibad_models.py, TrajectoryEncoder.__init__:")
    print(f"   Change: self.input_dim = state_dim + action_dim + 1")
    print(f"   To:     self.input_dim = {actual_from_trajectory}")
    print(f"   Or debug why the calculation gives {expected_by_encoder} instead of {actual_from_trajectory}")


if __name__ == "__main__":
    print("🔍 VariBAD Dimension Debug Tool")
    print("=" * 60)
    print("This script will help identify and fix the dimension mismatch error.")
    print("Run this before running your trainer to understand the issue.")
    print()
    
    # Run comprehensive debugging
    provide_fix_recommendations()
    
    print("\n" + "=" * 60)
    print("🎯 After running this debug script:")
    print("1. Update your varibad_models.py with the correct dimensions")
    print("2. Update your varibad_trainer.py with the correct parameters")
    print("3. Re-run the trainer with the fixed dimensions")
    print("=" * 60)