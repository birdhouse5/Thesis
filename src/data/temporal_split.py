"""Temporal split implementation for time series data."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TemporalSplitConfig:
    """Configuration for temporal split."""
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_method: str = 'continuous'
    buffer_days: int = 20
    episode_length: int = 252
    min_history: int = 60
    stride: int = 21
    task_boundaries: str = 'year'


class TemporalSplitter:
    """Split time series data temporally."""
    
    def __init__(self, config: TemporalSplitConfig):
        self.config = config
        self.splits = None
        
    def split_data(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: DataFrame with DatetimeIndex
            
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        total_days = len(data)
        
        if self.config.split_method == 'continuous':
            # Simple continuous split
            train_end = int(total_days * self.config.train_ratio)
            val_end = train_end + int(total_days * self.config.val_ratio)
            
            # Add buffer
            val_start = train_end + self.config.buffer_days
            test_start = val_end + self.config.buffer_days
            
            splits = {
                'train': data.iloc[:train_end],
                'val': data.iloc[val_start:val_end],
                'test': data.iloc[test_start:]
            }
            
        else:
            raise NotImplementedError(f"Split method {self.config.split_method} not implemented")
        
        # Log split information
        for name, split_data in splits.items():
            if len(split_data) > 0:
                logger.info(f"{name}: {len(split_data)} days, "
                          f"{split_data.index[0].date()} to {split_data.index[-1].date()}")
            else:
                logger.warning(f"{name}: Empty split!")
        
        self.splits = splits
        return splits
    
    def create_episodes(self, data: pd.DataFrame, 
                       split_name: str = 'train') -> List[Dict]:
        """
        Create episodes from a data split.
        
        Args:
            data: DataFrame with time series data
            split_name: Name of the split (for logging)
            
        Returns:
            List of episodes, each with start/end indices and task info
        """
        episodes = []
        
        # Need min_history before first episode
        start_idx = self.config.min_history
        
        while start_idx + self.config.episode_length <= len(data):
            end_idx = start_idx + self.config.episode_length
            
            episode = {
                'split': split_name,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_date': data.index[start_idx],
                'end_date': data.index[end_idx - 1],
                'history_start_idx': start_idx - self.config.min_history,
                'history_start_date': data.index[start_idx - self.config.min_history]
            }
            
            # Add task information
            if self.config.task_boundaries == 'year':
                episode['task_id'] = data.index[start_idx].year
            elif self.config.task_boundaries == 'quarter':
                episode['task_id'] = f"{data.index[start_idx].year}Q{data.index[start_idx].quarter}"
            else:
                episode['task_id'] = len(episodes)  # Just sequential
            
            episodes.append(episode)
            start_idx += self.config.stride
        
        logger.info(f"Created {len(episodes)} episodes for {split_name}")
        return episodes
    
    def create_all_episodes(self) -> Dict[str, List[Dict]]:
        """Create episodes for all splits."""
        if self.splits is None:
            raise ValueError("No splits available. Call split_data first.")
        
        all_episodes = {}
        
        for split_name, split_data in self.splits.items():
            if len(split_data) > self.config.min_history + self.config.episode_length:
                all_episodes[split_name] = self.create_episodes(split_data, split_name)
            else:
                logger.warning(f"Split {split_name} too small for episodes")
                all_episodes[split_name] = []
        
        return all_episodes
    
    def get_episode_data(self, data: pd.DataFrame, episode: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get data for a specific episode.
        
        Args:
            data: Full dataset
            episode: Episode dictionary
            
        Returns:
            Tuple of (history_data, episode_data)
        """
        history_data = data.iloc[episode['history_start_idx']:episode['start_idx']]
        episode_data = data.iloc[episode['start_idx']:episode['end_idx']]
        
        return history_data, episode_data
    
    def summarize_splits(self) -> pd.DataFrame:
        """Create summary of splits and episodes."""
        if self.splits is None:
            raise ValueError("No splits available. Call split_data first.")
        
        summary_data = []
        all_episodes = self.create_all_episodes()
        
        for split_name, split_data in self.splits.items():
            episodes = all_episodes.get(split_name, [])
            
            if len(split_data) > 0:
                summary_data.append({
                    'split': split_name,
                    'start_date': split_data.index[0],
                    'end_date': split_data.index[-1],
                    'total_days': len(split_data),
                    'num_episodes': len(episodes),
                    'unique_tasks': len(set(ep['task_id'] for ep in episodes)) if episodes else 0
                })
        
        return pd.DataFrame(summary_data)


def test_temporal_split():
    """Test temporal split functionality."""
    # Create sample data
    dates = pd.date_range('2018-01-01', '2023-12-31', freq='B')  # Business days
    data = pd.DataFrame(
        np.random.randn(len(dates), 3),
        index=dates,
        columns=['A', 'B', 'C']
    )
    
    print(f"Total data: {len(data)} days")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Create splitter
    config = TemporalSplitConfig()
    splitter = TemporalSplitter(config)
    
    # Split data
    splits = splitter.split_data(data)
    
    # Create episodes
    all_episodes = splitter.create_all_episodes()
    
    # Summary
    print("\nSplit Summary:")
    print(splitter.summarize_splits())
    
    # Test getting episode data
    if all_episodes['train']:
        first_episode = all_episodes['train'][0]
        history, episode_data = splitter.get_episode_data(data, first_episode)
        print(f"\nFirst training episode:")
        print(f"  History: {len(history)} days")
        print(f"  Episode: {len(episode_data)} days")
        print(f"  Task ID: {first_episode['task_id']}")
    
    return splitter


if __name__ == "__main__":
    test_temporal_split()