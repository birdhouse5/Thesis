import numpy as np

import logging
logger = logging.getLogger(__name__)


class Environment:
    def __init__(self, dataset, episode_length=60, num_assets=30):
        self.dataset = dataset
        self.episode_length = episode_length
        self.num_assets = num_assets
        self.current_step = 0
        self.episode_data = None
    
    def reset(self):
        """Sample new 60-day episode and return initial observation"""
        self._sample_episode()
        
        start_date = self.dataset.dates[self.episode_start_idx]
        end_date = self.dataset.dates[self.episode_start_idx + self.episode_length - 1]
        self.episode_info = {
            'start_idx': self.episode_start_idx,
            'start_date': start_date,
            'end_date': end_date
        }
        
        logger.info(f"Episode reset: {start_date} to {end_date}")
        
        return self.episode_data[0]
    
    def step(self, action):
        """Take action, return next_obs, reward, done, info"""
        # Discretize and normalize action
        portfolio_weights = self._discretize_action(action)
        
        # Calculate reward using current and next observations
        reward = self._calculate_reward(portfolio_weights)
        
        # Move to next timestep
        self.current_step += 1
        done = self.current_step >= self.episode_length - 1
        
        # Get next observation (if not done)
        next_obs = self.episode_data[self.current_step] if not done else None
        
        info = {'portfolio_weights': portfolio_weights}
        
        return next_obs, reward, done, info
    
    def _sample_episode(self):
        """Sample random 60-day window from data"""
        max_start_idx = len(self.dataset) - self.episode_length
        start_idx = np.random.randint(0, max_start_idx)
        end_idx = start_idx + self.episode_length
        
        self.episode_data = self.dataset.get_window(start_idx, end_idx)
        self.current_step = 0
    
    def _discretize_action(self, action_dict):
        """Convert hierarchical action to portfolio weights"""
        decisions = action_dict['decisions']        # Shape: (30, 3) - [long, short, neutral]
        long_weights = action_dict['long_weights']  # Shape: (30,)
        short_weights = action_dict['short_weights'] # Shape: (30,)
        
        # Get asset decisions (argmax or sample from categorical)
        asset_decisions = np.argmax(decisions, axis=1)  # 0=long, 1=short, 2=neutral
        
        # Create masks
        long_mask = (asset_decisions == 0)
        short_mask = (asset_decisions == 1)
        
        # Initialize portfolio weights
        portfolio_weights = np.zeros(self.num_assets)
        
        # Apply long weights (normalized among long positions)
        if long_mask.any():
            long_raw = long_weights[long_mask]
            long_normalized = long_raw / long_raw.sum()
            portfolio_weights[long_mask] = long_normalized
        
        # Apply short weights (normalized among short positions, made negative)
        if short_mask.any():
            short_raw = short_weights[short_mask]
            short_normalized = short_raw / short_raw.sum()
            portfolio_weights[short_mask] = -short_normalized
        
        # Log the action breakdown
        logger.debug(f"Assets: Long={long_mask.sum()}, Short={short_mask.sum()}, Neutral={(asset_decisions==2).sum()}")
        logger.debug(f"Portfolio weights sum: {portfolio_weights.sum():.3f}")
        
        return portfolio_weights