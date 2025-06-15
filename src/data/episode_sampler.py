import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

class EpisodeSampler:
    """Randomly samples fixed-length episodes from data"""
    
    def __init__(self, data: pd.DataFrame, episode_length: int = 60):
        self.data = data
        self.episode_length = episode_length
        self.valid_start_indices = self._compute_valid_starts()
        
    def _compute_valid_starts(self) -> List[int]:
        """Find all valid episode starting points"""
        # Episode can start from index 0 to len-episode_length
        max_start = len(self.data) - self.episode_length
        return list(range(max_start + 1))
    
    def sample_episode(self) -> pd.DataFrame:
        """Sample a random episode"""
        start_idx = np.random.choice(self.valid_start_indices)
        end_idx = start_idx + self.episode_length
        return self.data.iloc[start_idx:end_idx].copy()
    
    def sample_batch(self, batch_size: int) -> List[pd.DataFrame]:
        """Sample multiple episodes"""
        return [self.sample_episode() for _ in range(batch_size)]