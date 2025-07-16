"""
Trajectory Buffer for VariBAD Portfolio Optimization

Following the variBAD paper specifications:
- Store trajectory sequences τ:t = (s0, a0, r1, s1, a1, r2, ..., st)
- Construct sequences on-demand for encoder input
- Support episode-based sampling for p(M) distribution
- Memory-efficient storage with configurable buffer size

Key Design Principles:
1. Store raw (s, a, r, s') transitions and construct τ:t sequences on-demand
2. Episode-aware storage to respect task boundaries
3. Efficient sequence extraction for RNN encoder training
4. Support for sliding window sampling from historical data
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from collections import deque, namedtuple
import random

# Transition tuple for storing individual steps
Transition = namedtuple('Transition', [
    'state', 'action', 'reward', 'next_state', 'done', 'info'
])

# Episode metadata for trajectory construction
EpisodeInfo = namedtuple('EpisodeInfo', [
    'episode_id', 'start_idx', 'length', 'start_date', 'end_date'
])

class TrajectoryBuffer:
    """
    Trajectory buffer for VariBAD following the paper's specifications.
    
    The buffer stores raw transitions and constructs trajectory sequences τ:t 
    on-demand for the RNN encoder. This approach is memory-efficient and allows
    flexible sequence length extraction.
    """
    
    def __init__(self, 
                 max_buffer_size: int = 100000,
                 max_episodes: int = 1000,
                 device: str = 'cpu'):
        """
        Initialize trajectory buffer.
        
        Args:
            max_buffer_size: Maximum number of transitions to store
            max_episodes: Maximum number of episodes to track
            device: Device for tensor operations ('cpu' or 'cuda')
        """
        self.max_buffer_size = max_buffer_size
        self.max_episodes = max_episodes
        self.device = device
        
        # Storage for raw transitions
        self.transitions = deque(maxlen=max_buffer_size)
        
        # Episode tracking for trajectory construction
        self.episodes = deque(maxlen=max_episodes)
        self.current_episode_id = 0
        self.current_episode_start_idx = 0
        self.current_episode_length = 0
        
        # Statistics
        self.total_transitions = 0
        self.total_episodes = 0
        
    def start_episode(self, start_date: str = None, episode_info: Dict = None):
        """
        Signal the start of a new episode.
        
        Args:
            start_date: Date string for episode start (for p(M) tracking)
            episode_info: Additional episode metadata
        """
        # Finalize previous episode if needed
        if self.current_episode_length > 0:
            self._finalize_current_episode()
        
        # Start new episode
        self.current_episode_id += 1
        self.current_episode_start_idx = len(self.transitions)
        self.current_episode_length = 0
        
        # Store episode start info
        self._current_start_date = start_date
        self._current_episode_info = episode_info or {}
        
    def add_transition(self, 
                      state: np.ndarray, 
                      action: np.ndarray, 
                      reward: float, 
                      next_state: np.ndarray, 
                      done: bool, 
                      info: Dict = None):
        """
        Add a single transition to the buffer.
        
        Args:
            state: Current state s_t
            action: Action taken a_t  
            reward: Reward received r_{t+1}
            next_state: Next state s_{t+1}
            done: Episode termination flag
            info: Additional transition information
        """
        transition = Transition(
            state=state.copy() if isinstance(state, np.ndarray) else state,
            action=action.copy() if isinstance(action, np.ndarray) else action,
            reward=float(reward),
            next_state=next_state.copy() if isinstance(next_state, np.ndarray) else next_state,
            done=bool(done),
            info=info or {}
        )
        
        self.transitions.append(transition)
        self.current_episode_length += 1
        self.total_transitions += 1
        
        # Finalize episode if done
        if done:
            self._finalize_current_episode()
    
    def _finalize_current_episode(self):
        """Finalize the current episode and add to episode tracking."""
        if self.current_episode_length == 0:
            return
            
        episode_info = EpisodeInfo(
            episode_id=self.current_episode_id,
            start_idx=self.current_episode_start_idx,
            length=self.current_episode_length,
            start_date=getattr(self, '_current_start_date', None),
            end_date=self._current_episode_info.get('end_date', None)
        )
        
        self.episodes.append(episode_info)
        self.total_episodes += 1
        
        # Reset current episode tracking
        self.current_episode_length = 0
    
    def get_trajectory_sequence(self, 
                               episode_idx: int, 
                               sequence_length: int,
                               start_step: int = 0) -> Dict[str, torch.Tensor]:
        """
        Extract a trajectory sequence τ:t for encoder input.
        
        Following variBAD paper: τ:t = (s0, a0, r1, s1, a1, r2, ..., st)
        
        Args:
            episode_idx: Index of episode to sample from
            sequence_length: Length of sequence to extract (t+1)
            start_step: Starting step within episode
            
        Returns:
            Dictionary containing trajectory components as tensors
        """
        if episode_idx >= len(self.episodes):
            raise ValueError(f"Episode index {episode_idx} out of range")
        
        episode = self.episodes[episode_idx]
        
        # Validate sequence bounds
        max_length = min(sequence_length, episode.length - start_step)
        if max_length <= 0:
            raise ValueError(f"Invalid sequence parameters for episode {episode_idx}")
        
        # Extract transitions for this sequence
        episode_start = episode.start_idx
        seq_start = episode_start + start_step
        seq_end = seq_start + max_length
        
        # Handle buffer wraparound (circular buffer)
        transitions = []
        for i in range(seq_start, seq_end):
            buffer_idx = i % len(self.transitions) if len(self.transitions) == self.max_buffer_size else i
            if buffer_idx < len(self.transitions):
                transitions.append(self.transitions[buffer_idx])
        
        if not transitions:
            raise ValueError(f"No valid transitions found for sequence")
        
        # Construct trajectory following variBAD format: [s_t, a_t, r_{t+1}]
        states = []
        actions = []
        rewards = []
        
        for i, trans in enumerate(transitions):
            states.append(trans.state)
            actions.append(trans.action)
            rewards.append(trans.reward)
        
        # Convert to tensors
        trajectory = {
            'states': torch.FloatTensor(np.array(states)).to(self.device),
            'actions': torch.FloatTensor(np.array(actions)).to(self.device), 
            'rewards': torch.FloatTensor(rewards).to(self.device),
            'sequence_length': max_length,
            'episode_id': episode.episode_id,
            'start_step': start_step
        }
        
        return trajectory
    
    def sample_trajectories(self, 
                           batch_size: int, 
                           max_sequence_length: int,
                           min_sequence_length: int = 5) -> List[Dict[str, torch.Tensor]]:
        """
        Sample a batch of trajectory sequences for training.
        
        Args:
            batch_size: Number of sequences to sample
            max_sequence_length: Maximum sequence length
            min_sequence_length: Minimum sequence length
            
        Returns:
            List of trajectory dictionaries
        """
        if len(self.episodes) == 0:
            return []
        
        trajectories = []
        
        for _ in range(batch_size):
            # Sample random episode
            episode_idx = random.randint(0, len(self.episodes) - 1)
            episode = self.episodes[episode_idx]
            
            # Sample random sequence length and start position
            max_possible_length = min(max_sequence_length, episode.length)
            if max_possible_length < min_sequence_length:
                continue  # Skip episodes that are too short
                
            sequence_length = random.randint(min_sequence_length, max_possible_length)
            max_start_step = episode.length - sequence_length
            start_step = random.randint(0, max_start_step) if max_start_step > 0 else 0
            
            try:
                trajectory = self.get_trajectory_sequence(episode_idx, sequence_length, start_step)
                trajectories.append(trajectory)
            except ValueError:
                continue  # Skip invalid sequences
        
        return trajectories
    
    def get_episode_info(self, episode_idx: int) -> EpisodeInfo:
        """Get metadata for a specific episode."""
        if episode_idx >= len(self.episodes):
            raise ValueError(f"Episode index {episode_idx} out of range")
        return self.episodes[episode_idx]
    
    def get_full_episode(self, episode_idx: int) -> Dict[str, torch.Tensor]:
        """
        Get the complete trajectory for an episode.
        
        Args:
            episode_idx: Index of episode
            
        Returns:
            Complete episode trajectory
        """
        episode = self.get_episode_info(episode_idx)
        return self.get_trajectory_sequence(episode_idx, episode.length, 0)
    
    def clear(self):
        """Clear all stored data."""
        self.transitions.clear()
        self.episodes.clear()
        self.current_episode_id = 0
        self.current_episode_start_idx = 0
        self.current_episode_length = 0
        self.total_transitions = 0
        self.total_episodes = 0
    
    def __len__(self) -> int:
        """Return number of stored transitions."""
        return len(self.transitions)
    
    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        return {
            'total_transitions': self.total_transitions,
            'total_episodes': self.total_episodes,
            'current_transitions': len(self.transitions),
            'current_episodes': len(self.episodes),
            'buffer_utilization': len(self.transitions) / self.max_buffer_size,
            'episode_utilization': len(self.episodes) / self.max_episodes,
            'avg_episode_length': np.mean([ep.length for ep in self.episodes]) if self.episodes else 0
        }


# Utility functions for trajectory processing
def pad_trajectories(trajectories: List[Dict[str, torch.Tensor]], 
                    max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """
    Pad trajectories to the same length for batch processing.
    
    Args:
        trajectories: List of trajectory dictionaries
        max_length: Maximum length to pad to (if None, use longest sequence)
        
    Returns:
        Batched and padded trajectory tensors
    """
    if not trajectories:
        return {}
    
    # Determine padding length
    if max_length is None:
        max_length = max(traj['sequence_length'] for traj in trajectories)
    
    batch_size = len(trajectories)
    state_dim = trajectories[0]['states'].shape[1]
    action_dim = trajectories[0]['actions'].shape[1]
    
    # Initialize padded tensors
    device = trajectories[0]['states'].device
    padded_states = torch.zeros(batch_size, max_length, state_dim, device=device)
    padded_actions = torch.zeros(batch_size, max_length, action_dim, device=device)
    padded_rewards = torch.zeros(batch_size, max_length, device=device)
    sequence_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    # Fill padded tensors
    for i, traj in enumerate(trajectories):
        seq_len = traj['sequence_length']
        sequence_lengths[i] = seq_len
        
        padded_states[i, :seq_len] = traj['states']
        padded_actions[i, :seq_len] = traj['actions']  
        padded_rewards[i, :seq_len] = traj['rewards']
    
    return {
        'states': padded_states,
        'actions': padded_actions,
        'rewards': padded_rewards,
        'sequence_lengths': sequence_lengths,
        'batch_size': batch_size,
        'max_length': max_length
    }


def test_trajectory_buffer():
    """Test the trajectory buffer implementation."""
    print("Testing TrajectoryBuffer...")
    
    # Initialize buffer
    buffer = TrajectoryBuffer(max_buffer_size=1000, max_episodes=50)
    
    # Test episode 1
    buffer.start_episode(start_date="2020-01-01")
    
    for step in range(10):
        state = np.random.randn(20)  # Mock state
        action = np.random.randn(10)  # Mock action
        reward = np.random.randn()  # Mock reward
        next_state = np.random.randn(20)  # Mock next state
        done = (step == 9)  # Last step
        
        buffer.add_transition(state, action, reward, next_state, done)
    
    # Test episode 2
    buffer.start_episode(start_date="2020-01-02")
    
    for step in range(15):
        state = np.random.randn(20)
        action = np.random.randn(10)
        reward = np.random.randn()
        next_state = np.random.randn(20)
        done = (step == 14)
        
        buffer.add_transition(state, action, reward, next_state, done)
    
    # Test trajectory extraction
    print(f"Buffer stats: {buffer.get_stats()}")
    
    # Get trajectory from first episode
    trajectory = buffer.get_trajectory_sequence(episode_idx=0, sequence_length=5, start_step=2)
    print(f"Trajectory shapes: states {trajectory['states'].shape}, "
          f"actions {trajectory['actions'].shape}, rewards {trajectory['rewards'].shape}")
    
    # Test batch sampling
    batch = buffer.sample_trajectories(batch_size=3, max_sequence_length=8, min_sequence_length=3)
    print(f"Sampled batch size: {len(batch)}")
    
    # Test padding
    if batch:
        padded = pad_trajectories(batch)
        print(f"Padded batch shapes: states {padded['states'].shape}, "
              f"actions {padded['actions'].shape}, rewards {padded['rewards'].shape}")
    
    print("✅ TrajectoryBuffer tests passed!")


if __name__ == "__main__":
    test_trajectory_buffer()