
# ===========================================
# FILE 4: sp500_loader/core/complete_environment.py (COMPLETE IMPLEMENTATION)
# ===========================================

# complete_environment.py
"""
Complete Portfolio Management Environment - Research Paper Implementation
This is the full implementation with technical indicators and market features.
Use this for advanced research, use environment.py for basic functionality.
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, List
import warnings
warnings.filterwarnings('ignore')

from .environment import DifferentialSharpeRatio


class TechnicalIndicators:
    """Calculate technical indicators for asset features."""
    
    @staticmethod
    def sma(prices: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return prices.rolling(window=window, min_periods=1).mean()
    
    @staticmethod
    def ema(prices: pd.Series, span: int) -> pd.Series:
        """Exponential Moving Average"""
        return prices.ewm(span=span, min_periods=1).mean()
    
    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def momentum(prices: pd.Series, window: int = 10) -> pd.Series:
        """Price Momentum"""
        return prices / prices.shift(window) - 1


class CompletePortfolioEnv(gym.Env):
    """
    Complete Portfolio Environment with all research paper features.
    """
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        episode_length: int = 30,
        lookback_window: int = 20,
        initial_cash: float = 100000.0,
        short_ratio_mode: str = 'fixed',
        fixed_short_ratio: float = 0.0,
        max_short_ratio: float = 0.5,
        transaction_cost: float = 0.001,
        technical_indicators: List[str] = None,
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
        self.technical_indicators = technical_indicators or ['sma', 'rsi']
        
        # Extract basic info
        self.dates = sorted(price_data.index.get_level_values('date').unique())
        self.tickers = sorted(price_data.index.get_level_values('ticker').unique())
        self.n_assets = len(self.tickers)
        
        print(f"Complete Environment: {self.n_assets} assets, {len(self.technical_indicators)} indicators")
        
        # Calculate features
        self._calculate_features()
        
        # Initialize DSR
        default_dsr_config = {'decay_rate': 0.01}
        if dsr_config:
            default_dsr_config.update(dsr_config)
        self.dsr_calculator = DifferentialSharpeRatio(**default_dsr_config)
        
        # Action space
        if short_ratio_mode == 'fixed':
            action_dim = 2 * self.n_assets
        else:
            action_dim = 2 * self.n_assets + 1
            
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        
        # Observation space (larger due to technical indicators)
        n_tech_features = len(self.technical_indicators)
        obs_size = (self.lookback_window * self.n_assets * (1 + n_tech_features) +  # Asset features
                   5 +                                                              # Market features  
                   2 * self.n_assets)                                              # Account features
        
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
        
        print(f"  Action space: {self.action_space.shape}")
        print(f"  Observation space: {self.observation_space.shape}")
    
    def _calculate_features(self):
        """Calculate returns and technical indicators."""
        # Returns
        returns_data = []
        for ticker in self.tickers:
            ticker_prices = self.price_data.xs(ticker, level='ticker')['adj_close']
            ticker_returns = ticker_prices.pct_change().fillna(0)
            returns_data.append(ticker_returns)
        
        self.returns_df = pd.concat(returns_data, axis=1, keys=self.tickers)
        
        # Technical indicators
        self.tech_features = {}
        for ticker in self.tickers:
            ticker_prices = self.price_data.xs(ticker, level='ticker')['adj_close']
            features = {}
            
            if 'sma' in self.technical_indicators:
                features['sma'] = TechnicalIndicators.sma(ticker_prices, 20).fillna(0)
            if 'rsi' in self.technical_indicators:
                features['rsi'] = (TechnicalIndicators.rsi(ticker_prices) / 100.0).fillna(0.5)
            if 'momentum' in self.technical_indicators:
                features['momentum'] = TechnicalIndicators.momentum(ticker_prices).fillna(0)
            
            self.tech_features[ticker] = features
        
        # Market index (simple average of all stocks)
        self.market_index = self.returns_df.mean(axis=1)
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)
        
        min_start = self.lookback_window
        max_start = len(self.dates) - self.episode_length - 1
        
        if max_start <= min_start:
            raise ValueError(f"Not enough data")
        
        self.current_date_idx = np.random.randint(min_start, max_start)
        self.current_step = 0
        
        self.long_weights = np.ones(self.n_assets) / self.n_assets
        self.short_weights = np.zeros(self.n_assets)
        self.current_short_ratio = 0.0 if self.short_ratio_mode == 'learnable' else self.fixed_short_ratio
        self.portfolio_value = self.initial_cash
        self.episode_returns = []
        
        self.dsr_calculator.reset()
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute step."""
        # Normalize action (simplified)
        if self.short_ratio_mode == 'fixed':
            long_raw = action[:self.n_assets]
            short_raw = action[self.n_assets:]
            short_ratio = self.fixed_short_ratio
        else:
            long_raw = action[:self.n_assets]
            short_raw = action[self.n_assets:2*self.n_assets]
            short_ratio = np.clip(action[-1], 0.0, self.max_short_ratio)
        
        # Normalize weights
        long_weights = np.maximum(long_raw, 0)
        long_weights = long_weights / (long_weights.sum() + 1e-8)
        
        short_weights = np.minimum(short_raw, 0)
        if short_ratio > 0 and short_weights.sum() < 0:
            short_weights = short_weights * (short_ratio / abs(short_weights.sum()))
        else:
            short_weights = np.zeros(self.n_assets)
        
        self.long_weights = long_weights
        self.short_weights = short_weights
        self.current_short_ratio = short_ratio
        
        # Move forward
        self.current_date_idx += 1
        self.current_step += 1
        
        # Calculate return
        today_returns = self.returns_df.iloc[self.current_date_idx].values
        long_return = np.dot(self.long_weights, today_returns)
        short_return = -np.dot(np.abs(self.short_weights), today_returns)
        portfolio_return = long_return + short_return
        
        self.episode_returns.append(portfolio_return)
        self.portfolio_value *= (1 + portfolio_return)
        
        # DSR reward
        reward = self.dsr_calculator.update(portfolio_return)
        
        done = self.current_step >= self.episode_length
        
        info = {
            'portfolio_value': self.portfolio_value,
            'daily_return': portfolio_return,
            'long_weights': self.long_weights.copy(),
            'short_weights': self.short_weights.copy(),
            'short_ratio': self.current_short_ratio,
            'date': self.dates[self.current_date_idx],
            'step': self.current_step
        }
        
        if done:
            total_return = (self.portfolio_value / self.initial_cash) - 1
            info['episode_total_return'] = total_return
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get complete observation with technical indicators."""
        start_idx = max(0, self.current_date_idx - self.lookback_window)
        end_idx = self.current_date_idx
        
        # Asset features (returns + technical indicators)
        asset_features = []
        for ticker in self.tickers:
            # Returns
            returns_window = self.returns_df[ticker].iloc[start_idx:end_idx]
            if len(returns_window) < self.lookback_window:
                returns_window = np.pad(returns_window, (self.lookback_window - len(returns_window), 0))
            
            ticker_features = [returns_window]
            
            # Technical indicators
            for indicator in self.technical_indicators:
                if indicator in self.tech_features[ticker]:
                    tech_window = self.tech_features[ticker][indicator].iloc[start_idx:end_idx]
                    if len(tech_window) < self.lookback_window:
                        tech_window = np.pad(tech_window, (self.lookback_window - len(tech_window), 0))
                    ticker_features.append(tech_window)
            
            # Flatten features for this ticker
            asset_features.extend(np.concatenate(ticker_features))
        
        # Market features (simplified)
        market_return = self.market_index.iloc[self.current_date_idx] if self.current_date_idx < len(self.market_index) else 0
        market_features = [market_return, 0, 0, 0, 0]  # Simplified to 5 features
        
        # Account features
        account_features = np.concatenate([self.long_weights, self.short_weights])
        if self.short_ratio_mode == 'learnable':
            account_features = np.concatenate([account_features, [self.current_short_ratio]])
        
        # Combine all
        observation = np.concatenate([asset_features, market_features, account_features])
        return observation.astype(np.float32)


def create_complete_env_from_loader(loader, split='train', **env_kwargs):
    """Create complete environment from loader."""
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
    return CompletePortfolioEnv(multi_index_df, **env_kwargs)