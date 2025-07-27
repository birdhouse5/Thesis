"""
Utilities for VariBAD training
Consolidated from varibad/utils/buffer.py
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from collections import deque


class TrajectoryBuffer:
    """Buffer for storing trajectory sequences"""
    
    def __init__(self, max_episodes: int = 200, device: str = 'cpu'):
        self.max_episodes = max_episodes
        self.device = device
        self.episodes = deque(maxlen=max_episodes)
        self.current_sequence = []
    
    def start_sequence(self):
        """Start collecting a new sequence"""
        if self.current_sequence:
            self.episodes.append(self.current_sequence.copy())
        self.current_sequence = []
    
    def add_step(self, state: np.ndarray, action: np.ndarray, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Add a step to current sequence"""
        step = {
            'state': state.copy(),
            'action': action.copy(),
            'reward': reward,
            'next_state': next_state.copy(),
            'done': done
        }
        self.current_sequence.append(step)
        
        if done:
            self.episodes.append(self.current_sequence.copy())
            self.current_sequence = []
    
    def get_trajectory_sequence(self, episode_idx: int, sequence_length: int) -> Dict[str, torch.Tensor]:
        """Extract trajectory sequence for encoder"""
        if episode_idx >= len(self.episodes):
            raise ValueError(f"Episode {episode_idx} doesn't exist")
        
        transitions = self.episodes[episode_idx]
        actual_length = min(sequence_length, len(transitions))
        
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
        """Sample trajectory sequences for training"""
        if len(self.episodes) == 0:
            return []
        
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.randint(0, len(self.episodes))
            episode_length = len(self.episodes[episode_idx])
            seq_length = np.random.randint(1, min(max_seq_length, episode_length) + 1)
            
            try:
                trajectory = self.get_trajectory_sequence(episode_idx, seq_length)
                batch.append(trajectory)
            except (ValueError, IndexError):
                continue
        
        return batch
    
    def get_buffer_stats(self) -> Dict:
        """Get buffer statistics"""
        if not self.episodes:
            return {'num_sequences': 0, 'avg_length': 0}
        
        lengths = [len(ep) for ep in self.episodes]
        return {
            'num_sequences': len(self.episodes),
            'avg_length': np.mean(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'total_transitions': sum(lengths)
        }
    
    def clear(self):
        """Clear buffer"""
        self.episodes.clear()
        self.current_sequence = []


def create_trajectory_batch(trajectories: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Create padded batch from variable-length trajectories"""
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
    
    # Fill with data
    for i, traj in enumerate(trajectories):
        seq_len = traj['length']
        lengths[i] = seq_len
        
        states[i, :seq_len] = traj['states']
        actions[i, :seq_len] = traj['actions']
        rewards[i, :seq_len] = traj['rewards']
    
    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'lengths': lengths
    }


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_preference: str = 'auto') -> torch.device:
    """Get computing device"""
    if device_preference == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_preference)
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_training_stats(stats: Dict, filepath: str):
    """Save training statistics with proper JSON serialization"""
    import json
    
    # Convert numpy arrays to lists
    json_stats = {}
    for key, value in stats.items():
        if hasattr(value, 'tolist'):
            json_stats[key] = value.tolist()
        elif isinstance(value, (list, tuple)):
            json_stats[key] = list(value)
        else:
            json_stats[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(json_stats, f, indent=2)


def load_training_stats(filepath: str) -> Dict:
    """Load training statistics"""
    import json
    
    with open(filepath, 'r') as f:
        return json.load(f)