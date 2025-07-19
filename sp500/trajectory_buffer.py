"""
Blind Trajectory Buffer for Implicit Regime Detection

Key principles:
1. Agent receives ONLY raw time series data and features
2. NO metadata about market conditions, regimes, or episode context
3. Episodes are arbitrary training windows - agent must discover structure
4. Buffer simply stores sequential observations for variBAD encoder training
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from collections import deque

class BlindTrajectoryBuffer:
    """
    Trajectory buffer for implicit regime detection.
    
    Stores raw sequential observations without any regime/context metadata.
    The agent must discover market structure purely from the data patterns.
    """
    
    def __init__(self, max_episodes: int = 200, device: str = 'cpu'):
        """
        Args:
            max_episodes: Maximum number of training sequences to store
            device: Device for tensor operations
        """
        self.max_episodes = max_episodes
        self.device = device
        
        # Store complete sequences as raw transitions only
        self.episodes = deque(maxlen=max_episodes)
        
        # Current sequence being collected
        self.current_sequence = []
        
    def start_sequence(self):
        """Start collecting a new sequence (no metadata allowed)."""
        # Save previous sequence if it exists
        if self.current_sequence:
            self.episodes.append(self.current_sequence.copy())
        
        # Start fresh sequence
        self.current_sequence = []
    
    def add_step(self, state: np.ndarray, action: np.ndarray, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Add a single step - only raw data, no context information."""
        step = {
            'state': state.copy(),
            'action': action.copy(), 
            'reward': reward,
            'next_state': next_state.copy(),
            'done': done
        }
        self.current_sequence.append(step)
        
        # If sequence is complete, save it
        if done:
            self.episodes.append(self.current_sequence.copy())
            self.current_sequence = []
    
    def get_trajectory_sequence(self, episode_idx: int, sequence_length: int) -> Dict[str, torch.Tensor]:
        """
        Extract τ:t sequence for variBAD encoder.
        
        Returns only raw observations - no regime hints!
        """
        if episode_idx >= len(self.episodes):
            raise ValueError(f"Episode {episode_idx} doesn't exist")
        
        transitions = self.episodes[episode_idx]
        actual_length = min(sequence_length, len(transitions))
        
        # Extract raw observation sequences
        states = []
        actions = []
        rewards = []
        
        for i in range(actual_length):
            states.append(transitions[i]['state'])
            actions.append(transitions[i]['action'])
            rewards.append(transitions[i]['reward'])
        
        return {
            'states': torch.FloatTensor(np.array(states)).to(self.device),
            'actions': torch.FloatTensor(np.array(actions)).to(self.device),
            'rewards': torch.FloatTensor(rewards).to(self.device),
            'length': actual_length
        }
    
    def sample_training_batch(self, batch_size: int, max_seq_length: int) -> List[Dict[str, torch.Tensor]]:
        """
        Sample trajectory sequences for variBAD training.
        
        Each sequence represents a different "task" from p(M), but the agent
        doesn't know this - it just sees different data patterns.
        """
        if len(self.episodes) == 0:
            return []
        
        batch = []
        for _ in range(batch_size):
            # Sample random sequence from buffer
            episode_idx = np.random.randint(0, len(self.episodes))
            
            # Sample random sequence length
            episode_length = len(self.episodes[episode_idx])
            seq_length = np.random.randint(1, min(max_seq_length, episode_length) + 1)
            
            try:
                trajectory = self.get_trajectory_sequence(episode_idx, seq_length)
                batch.append(trajectory)
            except (ValueError, IndexError):
                continue
        
        return batch
    
    def get_buffer_stats(self) -> Dict:
        """Get basic buffer statistics (no regime information)."""
        if not self.episodes:
            return {'num_sequences': 0, 'avg_length': 0}
        
        lengths = [len(ep) for ep in self.episodes]
        
        return {
            'num_sequences': len(self.episodes),
            'avg_length': np.mean(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'total_transitions': sum(lengths),
            'buffer_usage': len(self.episodes) / self.max_episodes
        }
    
    def clear(self):
        """Clear all stored data."""
        self.episodes.clear()
        self.current_sequence = []


def create_trajectory_batch(trajectories: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Prepare trajectory batch for variBAD encoder training.
    
    Input format: List of variable-length sequences
    Output format: Padded tensors ready for RNN processing
    """
    if not trajectories:
        return {}
    
    # Find dimensions
    max_length = max(traj['length'] for traj in trajectories)
    batch_size = len(trajectories)
    state_dim = trajectories[0]['states'].shape[1]
    action_dim = trajectories[0]['actions'].shape[1]
    device = trajectories[0]['states'].device
    
    # Create padded tensors
    states = torch.zeros(batch_size, max_length, state_dim, device=device)
    actions = torch.zeros(batch_size, max_length, action_dim, device=device)
    rewards = torch.zeros(batch_size, max_length, device=device)
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    # Fill with trajectory data
    for i, traj in enumerate(trajectories):
        seq_len = traj['length']
        lengths[i] = seq_len
        
        states[i, :seq_len] = traj['states']
        actions[i, :seq_len] = traj['actions']
        rewards[i, :seq_len] = traj['rewards']
    
    return {
        'states': states,        # [batch_size, max_length, state_dim]
        'actions': actions,      # [batch_size, max_length, action_dim]  
        'rewards': rewards,      # [batch_size, max_length]
        'lengths': lengths       # [batch_size] - for RNN masking
    }


# How this connects to your portfolio MDP
def demonstrate_blind_collection():
    """
    Show how this works with your portfolio environment.
    
    Key insight: The buffer just collects raw sequences.
    The agent must figure out market patterns itself!
    """
    buffer = BlindTrajectoryBuffer()
    
    print("=== Simulating Portfolio Data Collection ===")
    print("Agent receives ONLY raw observations, no regime labels\n")
    
    # Simulate different market periods (but agent doesn't know this!)
    market_scenarios = [
        "Bull market (but agent doesn't know)",
        "Bear market (but agent doesn't know)", 
        "Volatile market (but agent doesn't know)"
    ]
    
    for scenario_idx, scenario_name in enumerate(market_scenarios):
        print(f"Collecting sequence {scenario_idx + 1}/3")
        print(f"True scenario: {scenario_name}")
        print("Agent sees: Just another sequence of states/actions/rewards\n")
        
        buffer.start_sequence()
        
        # Generate episode data (agent only sees the numbers!)
        for step in range(15):
            # Raw observations - no context clues about regime
            state = np.random.randn(50)      # Market features + indicators
            action = np.random.randn(20)     # Portfolio allocation
            reward = np.random.randn()       # DSR reward  
            next_state = np.random.randn(50) # Next observations
            done = (step == 14)
            
            buffer.add_step(state, action, reward, next_state, done)
    
    print("=== Buffer Contents ===")
    stats = buffer.get_buffer_stats()
    print(f"Stored {stats['num_sequences']} sequences")
    print(f"Average length: {stats['avg_length']:.1f} steps")
    print(f"Total transitions: {stats['total_transitions']}")
    
    print("\n=== What VariBAD Encoder Will See ===")
    batch = buffer.sample_training_batch(batch_size=2, max_seq_length=10)
    if batch:
        padded = create_trajectory_batch(batch)
        print(f"Batch shape: {padded['states'].shape}")
        print("Content: Raw state/action/reward sequences")
        print("Task: Learn that different sequences have different patterns")
        print("Goal: Encode these patterns into latent variable m")
    
    return buffer

if __name__ == "__main__":
    demonstrate_blind_collection()