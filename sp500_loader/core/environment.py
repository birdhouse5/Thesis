# ===========================================
# FILE 1: sp500_loader/core/environment.py (UPDATED - FIX IMPORTS)
# ===========================================

# environment.py
"""
Portfolio Management Gym Environment with Long-Short Capability and DSR Reward
Updated with proper imports and backward compatibility.
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any


class DifferentialSharpeRatio:
    """Differential Sharpe Ratio calculator."""
    
    def __init__(self, decay_rate: float = 0.01, risk_free_rate: float = 0.0, min_variance: float = 1e-8):
        self.decay_rate = decay_rate
        self.risk_free_rate = risk_free_rate
        self.min_variance = min_variance
        self.reset()
    
    def reset(self):
        self.alpha = None
        self.beta = None
        self.step_count = 0
        self.is_initialized = False
    
    def update(self, portfolio_return: float) -> float:
        excess_return = portfolio_return - self.risk_free_rate
        
        if not self.is_initialized:
            self.alpha = excess_return
            self.beta = excess_return**2
            self.is_initialized = True
            self.step_count = 1
            return 0.0
        
        # Calculate DSR
        delta_alpha = excess_return - self.alpha
        delta_beta = excess_return**2 - self.beta
        variance = self.beta - self.alpha**2
        
        if variance <= self.min_variance:
            dsr = 0.0
        else:
            numerator = self.beta * delta_alpha - 0.5 * self.alpha * delta_beta
            denominator = variance**(3/2)
            dsr = numerator / denominator if abs(denominator) > self.min_variance else 0.0
            dsr = np.clip(dsr, -10.0, 10.0)
        
        # Update moving averages
        self.alpha = self.alpha + self.decay_rate * delta_alpha
        self.beta = self.beta + self.decay_rate * delta_beta
        self.step_count += 1
        
        return dsr


class PortfolioEnv(gym.Env):
    """
    Portfolio environment with long-short capability and DSR reward.
    """
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        episode_length: int = 30,
        lookback_window: int = 5,
        initial_cash: float = 100000.0,
        short_ratio_mode: str = 'fixed',
        fixed_short_ratio: float = 0.0,
        max_short_ratio: float = 0.5,
        transaction_cost: float = 0.0,
        reward_mode: str = 'dsr',
        dsr_config: Optional[Dict] = None
    ):
        super().__init__()
        
        self.price_data = price_data
        self.episode_length = episode_length
        self.lookback_window = lookback_window
        self.initial_cash = initial_cash
        self.short_ratio_mode = short_ratio_mode
        self.fixed_short_ratio = fixed_short_ratio
        self.max_short_ratio = max_short_ratio
        self.transaction_cost = transaction_cost
        self.reward_mode = reward_mode
        
        # Extract basic info
        self.dates = sorted(price_data.index.get_level_values('date').unique())
        self.tickers = sorted(price_data.index.get_level_values('ticker').unique())
        self.n_assets = len(self.tickers)
        
        # Calculate returns
        print("Calculating returns...")
        self._calculate_returns()
        
        # Initialize DSR calculator if using DSR rewards
        if reward_mode == 'dsr':
            default_dsr_config = {'decay_rate': 0.01, 'risk_free_rate': 0.0}
            if dsr_config:
                default_dsr_config.update(dsr_config)
            self.dsr_calculator = DifferentialSharpeRatio(**default_dsr_config)
        else:
            self.dsr_calculator = None
        
        # Define action space
        if short_ratio_mode == 'fixed':
            action_dim = 2 * self.n_assets
        else:
            action_dim = 2 * self.n_assets + 1
            
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        
        # Define observation space
        obs_size = (self.lookback_window * self.n_assets +  # Past returns
                   self.n_assets +                          # Current long weights
                   self.n_assets)                           # Current short weights
        
        if short_ratio_mode == 'learnable':
            obs_size += 1
            
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        # Portfolio state
        self.current_step = 0
        self.current_date_idx = 0
        self.long_weights = None
        self.short_weights = None
        self.current_short_ratio = fixed_short_ratio
        self.portfolio_value = initial_cash
        self.episode_returns = []
        
        print(f"Portfolio Environment initialized:")
        print(f"  Assets: {self.n_assets}")
        print(f"  Reward mode: {reward_mode}")
        print(f"  Action space: {self.action_space.shape}")
        print(f"  Observation space: {self.observation_space.shape}")
    
    def _calculate_returns(self):
        """Pre-calculate returns for all assets."""
        returns_data = []
        for ticker in self.tickers:
            ticker_prices = self.price_data.xs(ticker, level='ticker')['adj_close']
            ticker_returns = ticker_prices.pct_change()
            returns_data.append(ticker_returns)
        
        self.returns_df = pd.concat(returns_data, axis=1, keys=self.tickers)
        self.returns_df = self.returns_df.fillna(0.0)
    
    def _normalize_action(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Normalize action to satisfy portfolio constraints."""
        if self.short_ratio_mode == 'fixed':
            long_raw = action[:self.n_assets]
            short_raw = action[self.n_assets:]
            short_ratio = self.fixed_short_ratio
        else:
            long_raw = action[:self.n_assets]
            short_raw = action[self.n_assets:2*self.n_assets]
            short_ratio = np.clip(action[-1], 0.0, self.max_short_ratio)
        
        # Normalize long weights
        long_raw = np.clip(long_raw, 0.0, 1.0)
        if long_raw.sum() > 0:
            long_weights = long_raw / long_raw.sum()
        else:
            long_weights = np.ones(self.n_assets) / self.n_assets
        
        # Normalize short weights
        short_raw = np.clip(short_raw, -1.0, 0.0)
        if short_ratio > 0 and short_raw.sum() < 0:
            short_weights = short_raw * (short_ratio / abs(short_raw.sum()))
        else:
            short_weights = np.zeros(self.n_assets)
            short_ratio = 0.0
        
        return long_weights, short_weights, short_ratio
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)
        
        min_start = self.lookback_window
        max_start = len(self.dates) - self.episode_length - 1
        
        if max_start <= min_start:
            raise ValueError(f"Not enough data: need {min_start + self.episode_length + 1} days, have {len(self.dates)}")
        
        self.current_date_idx = np.random.randint(min_start, max_start)
        self.current_step = 0
        
        self.long_weights = np.ones(self.n_assets) / self.n_assets
        self.short_weights = np.zeros(self.n_assets)
        self.current_short_ratio = 0.0 if self.short_ratio_mode == 'learnable' else self.fixed_short_ratio
        self.portfolio_value = self.initial_cash
        self.episode_returns = []
        
        if self.dsr_calculator:
            self.dsr_calculator.reset()
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute trading step."""
        long_weights, short_weights, short_ratio = self._normalize_action(action)
        
        # Transaction costs
        transaction_cost = 0.0
        if self.long_weights is not None and self.transaction_cost > 0:
            long_turnover = np.sum(np.abs(long_weights - self.long_weights))
            short_turnover = np.sum(np.abs(short_weights - self.short_weights))
            total_turnover = long_turnover + short_turnover
            transaction_cost = total_turnover * self.transaction_cost
        
        # Update portfolio weights
        self.long_weights = long_weights.copy()
        self.short_weights = short_weights.copy()
        self.current_short_ratio = short_ratio
        
        # Move to next day
        self.current_date_idx += 1
        self.current_step += 1
        
        # Get today's returns
        current_date = self.dates[self.current_date_idx]
        today_returns = self.returns_df.iloc[self.current_date_idx].values
        
        # Calculate portfolio return
        long_return = np.dot(self.long_weights, today_returns)
        short_return = -np.dot(np.abs(self.short_weights), today_returns)
        portfolio_return = long_return + short_return - transaction_cost
        
        self.episode_returns.append(portfolio_return)
        self.portfolio_value *= (1 + portfolio_return)
        
        # Calculate reward
        if self.reward_mode == 'dsr' and self.dsr_calculator:
            reward = self.dsr_calculator.update(portfolio_return)
        else:
            reward = portfolio_return
        
        done = self.current_step >= self.episode_length
        
        # Info dict
        info = {
            'portfolio_value': self.portfolio_value,
            'daily_return': portfolio_return,
            'long_return': long_return,
            'short_return': short_return,
            'transaction_cost': transaction_cost,
            'long_weights': self.long_weights.copy(),
            'short_weights': self.short_weights.copy(),
            'short_ratio': self.current_short_ratio,
            'date': current_date,
            'step': self.current_step,
            'net_leverage': 1.0 + short_ratio,
            'reward_mode': self.reward_mode
        }
        
        if self.reward_mode == 'dsr' and self.dsr_calculator:
            info.update({
                'dsr_reward': reward,
                'traditional_return': portfolio_return,
                'dsr_alpha': self.dsr_calculator.alpha,
                'dsr_beta': self.dsr_calculator.beta
            })
        
        if done:
            episode_returns_array = np.array(self.episode_returns)
            total_return = (self.portfolio_value / self.initial_cash) - 1
            traditional_sharpe = episode_returns_array.mean() / (episode_returns_array.std() + 1e-8)
            
            info.update({
                'episode_total_return': total_return,
                'episode_sharpe_ratio': traditional_sharpe,
                'episode_volatility': episode_returns_array.std(),
                'episode_length': len(self.episode_returns)
            })
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        start_idx = self.current_date_idx - self.lookback_window
        end_idx = self.current_date_idx
        
        if start_idx < 0:
            past_returns = np.zeros((self.lookback_window, self.n_assets))
            available_data = self.returns_df.iloc[0:end_idx].values
            past_returns[-len(available_data):] = available_data
        else:
            past_returns = self.returns_df.iloc[start_idx:end_idx].values
        
        observation_parts = [
            past_returns.flatten(),
            self.long_weights,
            self.short_weights
        ]
        
        if self.short_ratio_mode == 'learnable':
            observation_parts.append(np.array([self.current_short_ratio]))
        
        observation = np.concatenate(observation_parts)
        return observation.astype(np.float32)
    
    def render(self, mode='human'):
        """Render environment state."""
        if mode == 'human':
            print(f"Step {self.current_step}/{self.episode_length}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Short Ratio: {self.current_short_ratio:.2%}")


def create_env_from_loader(loader, split='train', **env_kwargs):
    """Create environment from QuickSplitLoader."""
    split_data = loader.splits['temporal_splits'][split]
    price_data = split_data['prices']
    
    price_df_list = []
    for date in price_data.index:
        for ticker in price_data.columns:
            if not pd.isna(price_data.loc[date, ticker]):
                price_df_list.append({
                    'date': date,
                    'ticker': ticker,
                    'adj_close': price_data.loc[date, ticker]
                })
    
    multi_index_df = pd.DataFrame(price_df_list).set_index(['date', 'ticker'])
    return PortfolioEnv(multi_index_df, **env_kwargs)


# Backward compatibility
MinimalPortfolioEnv = PortfolioEnv