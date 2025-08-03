import numpy as np
import logging
from logger_config import experiment_logger
import torch

logger = logging.getLogger(__name__)

class Environment:
    def __init__(self, dataset, episode_length=60, num_assets=30, dsr_eta=0.01):
        self.dataset = dataset
        self.episode_length = episode_length
        self.num_assets = num_assets
        self.current_step = 0
        self.episode_data = None
        self.episode_count = 0
    
        # DSR parameters
        self.dsr_eta = dsr_eta
        self.alpha_prev = 0.0  # EMA of returns
        self.beta_prev = 0.01  # EMA of squared returns (small initial value to avoid div by 0)
        self.prev_portfolio_value = None

    def _calculate_dsr(self, R_t, alpha_prev, beta_prev, eta):
        """
        Compute the differential Sharpe ratio (DSR) reward.
        Parameters:
        - R_t: float, return at time t
        - alpha_prev: float, previous EMA of returns (mean)
        - beta_prev: float, previous EMA of squared returns (second moment)
        - eta: float, decay rate (e.g., 0.01)
        Returns:
        - dsr: float, differential Sharpe ratio at time t
        - alpha_t: float, updated alpha
        - beta_t: float, updated beta
        """
        delta_alpha = R_t - alpha_prev
        delta_beta = R_t**2 - beta_prev
        alpha_t = alpha_prev + eta * delta_alpha
        beta_t = beta_prev + eta * delta_beta
        
        denom = (beta_prev - alpha_prev**2) ** 1.5
        if denom == 0 or denom < 1e-8:  # Avoid division by zero/very small numbers
            dsr = 0.0
        else:
            dsr = (beta_prev * delta_alpha - 0.5 * alpha_prev * delta_beta) / denom
        
        return dsr, alpha_t, beta_t

    def _calculate_portfolio_return(self, portfolio_weights):
        """Calculate portfolio return based on current and next observations"""
        if self.current_step >= len(self.episode_data) - 1:
            return 0.0  # No next observation available
        
        current_obs = self.episode_data[self.current_step]      # (num_assets, num_features)
        next_obs = self.episode_data[self.current_step + 1]     # (num_assets, num_features)
        
        # Find the index for close price (assuming 'close_norm' is in features)
        # You'll need to adjust this based on your actual feature columns
        close_idx = self._get_close_price_idx()
        
        current_prices = current_obs[:, close_idx]  # (num_assets,)
        next_prices = next_obs[:, close_idx]        # (num_assets,)
        
        # Calculate individual asset returns: (price_t+1 - price_t) / price_t
        asset_returns = (next_prices - current_prices) / (current_prices + 1e-8)  # Avoid div by 0
        
        # Calculate portfolio return as weighted sum
        portfolio_return = np.sum(portfolio_weights * asset_returns)
        
        return portfolio_return

    def _get_close_price_idx(self):
        """Get the index of close price in feature columns"""
        # This assumes 'close_norm' is in your selected features
        # You'll need to adjust based on your actual Dataset feature selection
        try:
            return self.dataset.feature_cols.index('close_norm')
        except ValueError:
            # Fallback: assume first feature is price-related
            logger.warning("'close_norm' not found in features, using index 0")
            return 0

    def _calculate_reward(self, portfolio_weights):
        """Calculate DSR reward for the given portfolio weights"""
        # Calculate portfolio return
        R_t = self._calculate_portfolio_return(portfolio_weights)
        
        # Calculate DSR reward
        dsr, alpha_t, beta_t = self._calculate_dsr(R_t, self.alpha_prev, self.beta_prev, self.dsr_eta)
        
        # Update state for next timestep
        self.alpha_prev = alpha_t
        self.beta_prev = beta_t
        
        # Log DSR components
        logger.debug(f"Step {self.current_step}: R_t={R_t:.6f}, DSR={dsr:.6f}, "
                    f"alpha={alpha_t:.6f}, beta={beta_t:.6f}")
        
        # Log to TensorBoard
        if experiment_logger:
            step_id = self.episode_count * self.episode_length + self.current_step
            experiment_logger.log_scalars('dsr/components', {
                'portfolio_return': R_t,
                'dsr_reward': dsr,
                'alpha': alpha_t,
                'beta': beta_t
            }, step_id)
        
        return dsr

    def reset(self):
        """Sample new 60-day episode and return initial observation"""
        self._sample_episode()
        
        # Reset DSR state for new episode
        self.alpha_prev = 0.0
        self.beta_prev = 0.01
        self.prev_portfolio_value = None
        
        # [Rest of existing reset code...]
        start_date = self.dataset.dates[self.episode_start_idx]
        end_date = self.dataset.dates[self.episode_start_idx + self.episode_length - 1]
        self.episode_info = {
            'start_idx': self.episode_start_idx,
            'start_date': start_date,
            'end_date': end_date,
            'episode_id': self.episode_count
        }
        
        logger.info(f"Episode {self.episode_count}: {start_date} to {end_date}")
        
        if experiment_logger:
            experiment_logger.log_scalar('environment/episode_start_idx', self.episode_start_idx, self.episode_count)
        
        self.episode_count += 1
        return self.episode_data[0]
    
    def step(self, action):
        """Take action, return next_obs, reward, done, info"""
        portfolio_weights = self._discretize_action(action)
        reward = self._calculate_reward(portfolio_weights)
        
        self.current_step += 1
        done = self.current_step >= self.episode_length - 1
        
        next_obs = self.episode_data[self.current_step] if not done else None
        
        # Log metrics
        if experiment_logger:
            step_id = self.episode_count * self.episode_length + self.current_step
            experiment_logger.log_scalar('environment/reward', reward, step_id)
            experiment_logger.log_histogram('environment/portfolio_weights', portfolio_weights, step_id)
            experiment_logger.log_scalar('environment/portfolio_concentration', 
                                        (portfolio_weights**2).sum(), step_id)  # Concentration metric
        
        info = {
            'portfolio_weights': portfolio_weights,
            'step': self.current_step,
            'episode_id': self.episode_info['episode_id']
        }
        
        return next_obs, reward, done, info
    
    def _sample_episode(self):
        """Sample random 60-day window from data"""
        max_start_idx = len(self.dataset) - self.episode_length
        self.episode_start_idx = np.random.randint(0, max_start_idx)
        end_idx = self.episode_start_idx + self.episode_length
        
        self.episode_data = self.dataset.get_window(self.episode_start_idx, end_idx)
        self.current_step = 0
    
    # In environments/env.py - make _discretize_action more robust
    def _discretize_action(self, action_dict):
        """Convert hierarchical action to portfolio weights"""
        decisions = action_dict['decisions']        
        long_weights = action_dict['long_weights']  
        short_weights = action_dict['short_weights'] 
        
        # Convert to numpy and handle both 1D and 2D cases
        if isinstance(decisions, torch.Tensor):
            decisions = decisions.detach().cpu().numpy()
            long_weights = long_weights.detach().cpu().numpy()
            short_weights = short_weights.detach().cpu().numpy()
        
        # Ensure we have the right shape - squeeze if batch dimension exists
        if decisions.ndim == 2:
            decisions = decisions.squeeze(0)  # Remove batch dim
            long_weights = long_weights.squeeze(0)
            short_weights = short_weights.squeeze(0)
        
        # Get asset decisions (0=long, 1=short, 2=neutral)
        asset_decisions = decisions  # Already integers from policy
        
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