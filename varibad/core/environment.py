"""
Base MetaTrader MDP implementation following the paper exactly.
Focus on core portfolio optimization logic without VariBAD extensions.
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional, Union
from collections import deque

class MetaTraderPortfolioMDP(gym.Env):
    """
    Base portfolio optimization environment following MetaTrader paper.
    
    State: s_t = {x_t^s; x_t^m; x_t^a}
    Action: a_t = [w_t^+; w_t^-] (long and short weights)  
    Reward: Differential Sharpe Ratio (DSR)
    """
    
    def __init__(self,
                 data: pd.DataFrame,
                 lookback_window: int = 20,
                 episode_length: int = 30,
                 initial_capital: float = 1000000.0,
                 transaction_cost: float = 0.001,
                 short_selling_enabled: bool = True,
                 max_short_ratio: float = 0.5,
                 risk_free_rate: float = 0.02):
        """
        Initialize MetaTrader portfolio environment.
        
        Args:
            data: Cleaned SP500 data with all features
            lookback_window: Days of price history for asset features  
            episode_length: Trading days per episode
            initial_capital: Starting portfolio value
            transaction_cost: Cost rate for rebalancing
            short_selling_enabled: Allow short positions
            max_short_ratio: Maximum short position ratio
            risk_free_rate: Annual risk-free rate for DSR
        """
        super().__init__()
        
        # Store configuration
        self.data = data.sort_values(['date', 'ticker']).reset_index(drop=True)
        self.lookback_window = lookback_window
        self.episode_length = episode_length
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.short_selling_enabled = short_selling_enabled
        self.max_short_ratio = max_short_ratio
        self.risk_free_rate = risk_free_rate / 252  # Convert to daily
        
        # Extract unique dates and tickers
        self.dates = sorted(self.data['date'].unique())
        self.tickers = sorted(self.data['ticker'].unique())
        self.n_assets = len(self.tickers)
        
        print(f"MetaTrader MDP initialized:")
        print(f"  Assets: {self.n_assets}")
        print(f"  Date range: {self.dates[0].date()} to {self.dates[-1].date()}")
        print(f"  Total trading days: {len(self.dates)}")
        print(f"  Episode length: {self.episode_length}")
        print(f"  Short selling: {self.short_selling_enabled}")
        
        # Identify feature columns
        self._identify_feature_columns()
        
        # Define action and observation spaces
        self._define_spaces()
        
        # Initialize episode state
        self.reset()
    
    def _identify_feature_columns(self):
        """Identify different types of features in the dataset."""
        all_cols = set(self.data.columns)
        
        # Basic columns
        basic_cols = {'date', 'ticker'}
        
        # Asset features (x_t^s): normalized technical indicators + price features
        asset_feature_cols = [col for col in all_cols 
                             if col.endswith('_norm') or col in ['returns', 'log_returns']]
        
        # Market features (x_t^m): market-wide statistics
        market_feature_cols = ['market_return', 'volatility_5d', 'volatility_20d', 'excess_returns']
        market_feature_cols = [col for col in market_feature_cols if col in all_cols]
        
        self.asset_feature_cols = asset_feature_cols
        self.market_feature_cols = market_feature_cols
        
        print(f"Feature identification:")
        print(f"  Asset features: {len(asset_feature_cols)} columns")
        print(f"  Market features: {len(market_feature_cols)} columns")
        print(f"  Example asset features: {asset_feature_cols[:5]}")
        print(f"  Market features: {market_feature_cols}")
    
    def _define_spaces(self):
        """Define gym action and observation spaces."""
        
        # Action space: [w_t^+; w_t^-] 
        if self.short_selling_enabled:
            # 2N-dimensional: N long weights + N short weights
            action_dim = 2 * self.n_assets
            # Long weights: [0, 1], Short weights: [-max_short_ratio, 0]
            action_low = np.concatenate([
                np.zeros(self.n_assets),  # Long weights >= 0
                np.full(self.n_assets, -self.max_short_ratio)  # Short weights <= 0
            ])
            action_high = np.concatenate([
                np.ones(self.n_assets),   # Long weights <= 1
                np.zeros(self.n_assets)   # Short weights >= -max_short_ratio
            ])
        else:
            # N-dimensional: only long weights
            action_dim = self.n_assets
            action_low = np.zeros(self.n_assets)
            action_high = np.ones(self.n_assets)
        
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            dtype=np.float32
        )
        
        # Observation space: s_t = {x_t^s; x_t^m; x_t^a}
        asset_feature_dim = len(self.asset_feature_cols) * self.n_assets  # Per-asset features
        market_feature_dim = len(self.market_feature_cols)  # Market-wide features
        account_feature_dim = action_dim  # Previous portfolio weights
        
        obs_dim = asset_feature_dim + market_feature_dim + account_feature_dim
        
        # Observation bounds (normalized features should be reasonable)
        obs_low = np.full(obs_dim, -10.0)  # Conservative bounds for normalized data
        obs_high = np.full(obs_dim, 10.0)
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high, 
            dtype=np.float32
        )
        
        print(f"Space definition:")
        print(f"  Action space: {self.action_space.shape}")
        print(f"  Observation space: {self.observation_space.shape}")
        print(f"    Asset features: {asset_feature_dim}")
        print(f"    Market features: {market_feature_dim}")
        print(f"    Account features: {account_feature_dim}")
    
    def _sample_episode_start(self) -> int:
        """Sample a valid episode start date."""
        # Need enough history for lookback + full episode
        min_start_idx = self.lookback_window
        max_start_idx = len(self.dates) - self.episode_length - 1
        
        if min_start_idx >= max_start_idx:
            raise ValueError(f"Not enough data: need {self.lookback_window + self.episode_length} days")
        
        start_idx = np.random.randint(min_start_idx, max_start_idx)
        return start_idx
    
    def _get_date_data(self, date_idx: int) -> pd.DataFrame:
        """Get all ticker data for a specific date index."""
        date = self.dates[date_idx]
        date_data = self.data[self.data['date'] == date].copy()
        
        # Ensure we have data for all tickers (fill missing with previous day if needed)
        if len(date_data) < self.n_assets:
            print(f"Warning: Missing data for some tickers on {date.date()}")
            
        return date_data.set_index('ticker').reindex(self.tickers)
    
    def _construct_state(self, date_idx: int, previous_weights: np.ndarray) -> np.ndarray:
        """
        Construct state s_t = {x_t^s; x_t^m; x_t^a} following MetaTrader formulation.
        
        Args:
            date_idx: Current date index
            previous_weights: Portfolio weights from previous timestep
            
        Returns:
            State vector combining asset, market, and account features
        """
        
        # Get current date data
        current_data = self._get_date_data(date_idx)
        
        # Asset features (x_t^s): per-asset technical indicators
        asset_features = []
        for ticker in self.tickers:
            if ticker in current_data.index:
                ticker_features = current_data.loc[ticker, self.asset_feature_cols].values
                # Handle any remaining NaNs
                ticker_features = np.nan_to_num(ticker_features, nan=0.0)
            else:
                # Missing ticker data - use zeros
                ticker_features = np.zeros(len(self.asset_feature_cols))
            asset_features.append(ticker_features)
        
        asset_features = np.concatenate(asset_features)
        
        # Market features (x_t^m): market-wide indicators
        # Take first ticker's market features (same across all tickers)
        if len(current_data) > 0:
            market_features = current_data.iloc[0][self.market_feature_cols].values
            market_features = np.nan_to_num(market_features, nan=0.0)
        else:
            market_features = np.zeros(len(self.market_feature_cols))
        
        # Account features (x_t^a): previous portfolio weights
        account_features = previous_weights.copy()
        
        # Combine all features
        state = np.concatenate([asset_features, market_features, account_features])
        
        return state.astype(np.float32)
    
    def _validate_action(self, action: np.ndarray) -> np.ndarray:
        """
        Validate and normalize action following MetaTrader constraints:
        
        MetaTrader Portfolio Logic:
        - Long weights: w_i^+ ∈ [0,1], ∑w_i^+ = 1 (100% of capital invested long)
        - Short weights: w_i^- ∈ [-1,0], ∑w_i^- = -ρ_t (ρ_t = short ratio)
        - No simultaneous long/short positions on same asset
        - Net position per asset: w_i^+ + w_i^- (can be positive, negative, or zero)
        
        Example:
        - Long: [0.6, 0.4, 0.0] → sum = 1.0 (fully invested)
        - Short: [0.0, 0.0, -0.3] → sum = -0.3 (30% short ratio)
        - Net: [0.6, 0.4, -0.3] → Gross exposure = 130%, Net exposure = 70%
        """
        action = np.array(action, dtype=np.float32)
        
        if self.short_selling_enabled:
            # Split into long and short components
            long_weights = action[:self.n_assets].copy()
            short_weights = action[self.n_assets:].copy()
            
            # Constraint 1: No simultaneous long/short positions
            for i in range(self.n_assets):
                if long_weights[i] > 1e-8 and short_weights[i] < -1e-8:
                    # Keep the larger absolute position
                    if abs(long_weights[i]) >= abs(short_weights[i]):
                        short_weights[i] = 0.0  # Keep long, remove short
                    else:
                        long_weights[i] = 0.0   # Keep short, remove long
            
            # Constraint 2: Long weights sum to 1 (100% capital allocation)
            long_weights = np.maximum(long_weights, 0.0)  # Ensure non-negative
            long_sum = np.sum(long_weights)
            if long_sum > 1e-8:
                long_weights = long_weights / long_sum  # Normalize to sum=1
            else:
                # If no long positions specified, equal weight allocation
                long_weights = np.ones(self.n_assets) / self.n_assets
            
            # Constraint 3: Short weights respect max_short_ratio
            short_weights = np.minimum(short_weights, 0.0)  # Ensure non-positive
            short_sum = np.sum(short_weights)  # This will be ≤ 0
            if short_sum < -1e-8:
                # Current short ratio (absolute value)
                current_short_ratio = abs(short_sum)
                if current_short_ratio > self.max_short_ratio:
                    # Scale down to respect limit: keep proportions but reduce magnitude
                    scale_factor = self.max_short_ratio / current_short_ratio
                    short_weights = short_weights * scale_factor
                # If within limit, keep as is
            # If short_sum >= 0, no short positions, keep zeros
            
            # Final validation
            long_weights = np.clip(long_weights, 0.0, 1.0)
            short_weights = np.clip(short_weights, -self.max_short_ratio, 0.0)
            
            # Ensure long weights still sum to 1 after clipping
            if np.sum(long_weights) > 0:
                long_weights = long_weights / np.sum(long_weights)
            else:
                long_weights = np.ones(self.n_assets) / self.n_assets
            
            normalized_action = np.concatenate([long_weights, short_weights])
            
            # Verification (for debugging)
            final_long_sum = np.sum(long_weights)
            final_short_sum = np.sum(short_weights)
            assert abs(final_long_sum - 1.0) < 1e-6, f"Long weights sum = {final_long_sum}, should be 1.0"
            assert final_short_sum >= -self.max_short_ratio - 1e-6, f"Short ratio = {abs(final_short_sum)}, max = {self.max_short_ratio}"
            
        else:
            # Long-only: simple case, just normalize to sum to 1
            normalized_action = np.maximum(action, 0.0)  # Ensure non-negative
            action_sum = np.sum(normalized_action)
            if action_sum > 1e-8:
                normalized_action = normalized_action / action_sum
            else:
                normalized_action = np.ones(self.n_assets) / self.n_assets
            
            # Final bounds and normalization
            normalized_action = np.clip(normalized_action, 0.0, 1.0)
            normalized_action = normalized_action / np.sum(normalized_action)
        
        return normalized_action
    
    def _execute_action(self, action: np.ndarray, date_idx: int) -> Tuple[float, float]:
        """
        Execute portfolio rebalancing following MetaTrader logic.
        
        MetaTrader Portfolio Execution:
        1. Long positions: Invest 100% of capital according to w^+ weights
        2. Short positions: Borrow and sell according to w^- weights (additional capital)
        3. Net return: Weighted combination of long and short returns
        4. Transaction costs: Based on total weight changes
        
        Args:
            action: Validated portfolio weights [w^+; w^-]
            date_idx: Current date index
            
        Returns:
            (portfolio_return, transaction_cost)
        """
        
        # Get return data for current date
        current_data = self._get_date_data(date_idx)
        
        if self.short_selling_enabled:
            long_weights = action[:self.n_assets]
            short_weights = action[self.n_assets:]
            
            # Calculate returns for long and short positions separately
            long_return = 0.0
            short_return = 0.0
            valid_assets = 0
            
            for i, ticker in enumerate(self.tickers):
                if ticker in current_data.index:
                    asset_return = current_data.loc[ticker, 'returns']
                    if not np.isnan(asset_return):
                        # Long position contributes positively
                        long_return += long_weights[i] * asset_return
                        # Short position contributes negatively (profit when price falls)
                        short_return += short_weights[i] * asset_return  # short_weights[i] ≤ 0
                        valid_assets += 1
            
            # Total portfolio return = long return + short return
            # Note: short_return is ≤ 0 when prices rise (short loss)
            #       short_return is ≥ 0 when prices fall (short gain)
            portfolio_return = long_return + short_return
            
            # Current net positions for transaction cost calculation
            current_net_weights = long_weights + short_weights
            
        else:
            # Long-only case
            portfolio_return = 0.0
            valid_assets = 0
            
            for i, ticker in enumerate(self.tickers):
                if ticker in current_data.index:
                    asset_return = current_data.loc[ticker, 'returns']
                    if not np.isnan(asset_return):
                        portfolio_return += action[i] * asset_return
                        valid_assets += 1
            
            current_net_weights = action.copy()
        
        # Calculate transaction costs based on net position changes
        if hasattr(self, 'previous_net_weights'):
            weight_changes = np.abs(current_net_weights - self.previous_net_weights)
            transaction_cost = np.sum(weight_changes) * self.transaction_cost
        else:
            # First step: assume starting from equal weights
            if self.short_selling_enabled:
                initial_weights = np.concatenate([
                    np.ones(self.n_assets) / self.n_assets,  # Equal long
                    np.zeros(self.n_assets)  # No short
                ])
                initial_net_weights = initial_weights[:self.n_assets] + initial_weights[self.n_assets:]
            else:
                initial_net_weights = np.ones(self.n_assets) / self.n_assets
            
            weight_changes = np.abs(current_net_weights - initial_net_weights)
            transaction_cost = np.sum(weight_changes) * self.transaction_cost
        
        # Net return after transaction costs
        net_return = portfolio_return - transaction_cost
        
        # Update tracking variables
        if self.short_selling_enabled:
            self.current_weights = action.copy()  # Store full action [long; short]
        else:
            self.current_weights = current_net_weights.copy()
        
        self.previous_net_weights = current_net_weights.copy()
        
        return net_return, transaction_cost
    
    def _calculate_dsr_reward(self, return_rate: float) -> float:
        """
        Calculate Differential Sharpe Ratio (DSR) reward following MetaTrader paper.
        
        DSR_t = (β_{t-1} * Δα_t - 0.5 * α_{t-1} * Δβ_t) / (β_{t-1} - α_{t-1}²)^{3/2}
        
        where α_t and β_t are exponential moving estimates of first and second moments.
        
        Clipping rationale:
        - Early episodes: estimates are unstable → extreme DSR values
        - Numerical issues: denominator near zero → infinite DSR
        - RL training: extreme rewards destabilize learning
        - Range [-10, 10] keeps rewards meaningful but bounded
        """
        
        # Update exponential moving averages
        eta = 0.01  # Decay rate (as in MetaTrader paper)
        excess_return = return_rate - self.risk_free_rate
        
        # Update first moment (mean)
        delta_alpha = excess_return - self.alpha
        self.alpha = self.alpha + eta * delta_alpha
        
        # Update second moment (variance estimate)  
        delta_beta = excess_return**2 - self.beta
        self.beta = self.beta + eta * delta_beta
        
        # Calculate DSR with numerical stability checks
        variance_estimate = self.beta - self.alpha**2
        
        # Avoid division by zero or very small denominators
        if variance_estimate > 1e-6:  # Minimum variance threshold
            denominator = variance_estimate**(3/2)
            numerator = self.beta * delta_alpha - 0.5 * self.alpha * delta_beta
            dsr = numerator / denominator
        else:
            # Fallback: use simple return-based reward when variance estimate is too small
            dsr = excess_return * 10  # Scale to similar magnitude as typical DSR
        
        # Clip to prevent extreme values that destabilize RL training
        # Range chosen based on typical DSR magnitudes in financial literature
        dsr = np.clip(dsr, -10.0, 10.0)
        
        return dsr
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment and return initial observation."""
        if seed is not None:
            np.random.seed(seed)
        
        # Sample episode start
        self.episode_start_idx = self._sample_episode_start()
        self.current_step = 0
        
        # Initialize portfolio (equal weights or cash)
        if self.short_selling_enabled:
            # Start with equal long weights, no short positions
            initial_weights = np.concatenate([
                np.ones(self.n_assets) / self.n_assets,  # Equal long weights
                np.zeros(self.n_assets)  # No short positions
            ])
        else:
            initial_weights = np.ones(self.n_assets) / self.n_assets
        
        self.current_weights = initial_weights.copy()
        
        # Initialize DSR tracking
        self.alpha = 0.0  # First moment estimate
        self.beta = 0.01  # Second moment estimate (small positive value)
        
        # Construct initial state
        current_date_idx = self.episode_start_idx + self.current_step
        state = self._construct_state(current_date_idx, self.current_weights)
        
        # Initialize episode tracking
        self.episode_returns = []
        self.episode_actions = []
        self.episode_states = []
        
        return state
    
    def step(self, action: Union[np.ndarray, List]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Portfolio weights [w_t^+; w_t^-]
            
        Returns:
            (next_state, reward, done, info)
        """
        
        # Validate and normalize action
        action = self._validate_action(np.array(action))
        
        # Execute action and get return
        current_date_idx = self.episode_start_idx + self.current_step
        portfolio_return, transaction_cost = self._execute_action(action, current_date_idx + 1)
        
        # Calculate DSR reward
        reward = self._calculate_dsr_reward(portfolio_return)
        
        # Update step counter
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.episode_length
        
        # Construct next state (if not done)
        if not done:
            next_date_idx = self.episode_start_idx + self.current_step
            next_state = self._construct_state(next_date_idx, self.current_weights)
        else:
            next_state = np.zeros_like(self.observation_space.sample())
        
        # Store episode data
        self.episode_returns.append(portfolio_return)
        self.episode_actions.append(action.copy())
        
        # Info dictionary
        info = {
            'portfolio_return': portfolio_return,
            'transaction_cost': transaction_cost,
            'net_return': portfolio_return - transaction_cost,
            'dsr_reward': reward,
            'current_weights': self.current_weights.copy(),
            'date': self.dates[current_date_idx + 1].strftime('%Y-%m-%d'),
            'step': self.current_step,
            'alpha': self.alpha,
            'beta': self.beta
        }
        
        return next_state, reward, done, info
    
    def render(self, mode='human'):
        """Render environment state (optional)."""
        if mode == 'human':
            current_date = self.dates[self.episode_start_idx + self.current_step]
            print(f"Date: {current_date.date()}, Step: {self.current_step}/{self.episode_length}")
            print(f"Portfolio weights: {self.current_weights}")
            if self.episode_returns:
                print(f"Last return: {self.episode_returns[-1]:.6f}")

# Test function with detailed constraint validation
def test_metatrader_mdp():
    """Test the MetaTrader MDP implementation with constraint validation."""
    print("Testing MetaTrader MDP with Portfolio Constraints...")
    
    # Load cleaned data
    try:
        data = pd.read_parquet('data/sp500_rl_ready_cleaned.parquet')
        print(f"✓ Data loaded: {data.shape}")
    except FileNotFoundError:
        print("❌ Please run the NaN cleaning first to create sp500_rl_ready_cleaned.parquet")
        return None
    
    # Test 1: Long-only environment
    print(f"\n=== TEST 1: LONG-ONLY ENVIRONMENT ===")
    env_long = MetaTraderPortfolioMDP(
        data=data,
        episode_length=10,
        short_selling_enabled=False
    )
    
    obs = env_long.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial weights sum: {env_long.current_weights.sum():.6f} (should be 1.0)")
    
    # Test action validation - long only
    test_action_long = np.array([0.3, 0.2, 0.1, 0.4] + [0.0] * (env_long.n_assets - 4))
    validated = env_long._validate_action(test_action_long)
    print(f"Test action: {test_action_long[:4]}")
    print(f"Validated action sum: {validated.sum():.6f} (should be 1.0)")
    
    # Test 2: Short-selling environment
    print(f"\n=== TEST 2: SHORT-SELLING ENVIRONMENT ===")
    env_short = MetaTraderPortfolioMDP(
        data=data,
        episode_length=10,
        short_selling_enabled=True,
        max_short_ratio=0.3
    )
    
    obs = env_short.reset(seed=42)
    print(f"Action space shape: {env_short.action_space.shape} (should be {2 * env_short.n_assets})")
    
    # Test constraint validation with REALISTIC actions
    print(f"\n--- Realistic Constraint Validation Tests ---")
    
    # Test case 1: Realistic mixed long/short (no conflicts)
    n_assets = env_short.n_assets
    test_action = np.concatenate([
        # Long positions: concentrate in first few assets
        [0.3, 0.3, 0.2, 0.2] + [0.0] * (n_assets - 4),  
        # Short positions: different assets, realistic amounts
        [0.0] * 4 + [-0.1, -0.1, -0.1] + [0.0] * (n_assets - 7)
    ])
    
    validated = env_short._validate_action(test_action)
    long_part = validated[:n_assets]
    short_part = validated[n_assets:]
    
    print(f"Test 1 - Realistic mixed portfolio:")
    print(f"  Long sum: {long_part.sum():.6f} (should be 1.0)")
    print(f"  Short sum: {short_part.sum():.6f} (should be ~-0.3)")
    print(f"  Net exposure: {(long_part.sum() + short_part.sum()):.6f}")
    print(f"  No conflicts: {all(not (long_part[i] > 0 and short_part[i] < 0) for i in range(n_assets))}")
    
    # Test case 2: Aggressive short strategy at limit
    test_action_aggressive = np.concatenate([
        # Long: 50-50 split in first two assets
        [0.5, 0.5] + [0.0] * (n_assets - 2),  
        # Short: Exactly at 30% limit, spread across multiple assets
        [0.0, 0.0] + [-0.1, -0.1, -0.1] + [0.0] * (n_assets - 5)
    ])
    
    validated_aggressive = env_short._validate_action(test_action_aggressive)
    long_part = validated_aggressive[:n_assets]
    short_part = validated_aggressive[n_assets:]
    
    print(f"\nTest 2 - Aggressive short strategy:")
    print(f"  Long sum: {long_part.sum():.6f}")
    print(f"  Short sum: {short_part.sum():.6f} (should be ~-0.3)")
    print(f"  Gross exposure: {(long_part.sum() + abs(short_part.sum())):.6f} (130% leverage)")
    print(f"  Net exposure: {(long_part.sum() + short_part.sum()):.6f} (70% net)")
    
    # Test case 3: Scaling down excessive shorts
    test_action_excessive = np.concatenate([
        # Long: Equal weight in first 3 assets
        [0.33, 0.33, 0.34] + [0.0] * (n_assets - 3),
        # Short: 50% total (exceeds 30% limit) - should be scaled down
        [0.0] * 3 + [-0.2, -0.15, -0.15] + [0.0] * (n_assets - 6)
    ])
    
    validated_excessive = env_short._validate_action(test_action_excessive)
    short_part = validated_excessive[n_assets:]
    original_short_sum = np.sum(test_action_excessive[n_assets:])
    
    print(f"\nTest 3 - Excessive shorts (scaling test):")
    print(f"  Original short sum: {original_short_sum:.6f} (-50%)")
    print(f"  Scaled short sum: {short_part.sum():.6f} (should be ~-30%)")
    print(f"  Scale factor applied: {(short_part.sum() / original_short_sum):.6f} (should be ~0.6)")
    
    # Test 3: Episode execution with realistic actions
    print(f"\n=== TEST 3: EPISODE EXECUTION ===")
    
    total_reward = 0
    for step in range(5):
        # Create realistic portfolio action (no conflicts)
        # Strategy: Long positions in first half of assets, short in second half
        n_long = n_assets // 2
        n_short = n_assets - n_long
        
        # Generate random but realistic allocation
        long_weights = np.random.dirichlet(np.ones(n_long))
        long_action = np.zeros(n_assets)
        long_action[:n_long] = long_weights
        
        # Random short positions on different assets (small amounts)
        short_weights = np.random.uniform(-0.05, 0, n_short)  # Small short positions
        short_action = np.zeros(n_assets)  
        short_action[n_long:] = short_weights
        
        # Combine into full action
        action = np.concatenate([long_action, short_action])
        
        obs, reward, done, info = env_short.step(action)
        total_reward += reward
        
        # Calculate actual portfolio metrics
        current_long = info['current_weights'][:n_assets]
        current_short = info['current_weights'][n_assets:]
        net_weights = current_long + current_short
        gross_exposure = np.sum(np.abs(current_long)) + np.sum(np.abs(current_short))
        net_exposure = np.sum(net_weights)
        
        print(f"Step {step+1}:")
        print(f"  Portfolio return: {info['portfolio_return']:.6f}")
        print(f"  Transaction cost: {info['transaction_cost']:.6f}")
        print(f"  DSR reward: {info['dsr_reward']:.6f}")
        print(f"  Long sum: {current_long.sum():.6f}")
        print(f"  Short sum: {current_short.sum():.6f}")
        print(f"  Gross exposure: {gross_exposure:.6f} (leverage)")
        print(f"  Net exposure: {net_exposure:.6f}")
        
        if done:
            break
    
    print(f"\nEpisode summary:")
    print(f"  Total DSR reward: {total_reward:.6f}")
    print(f"  Average return: {np.mean(env_short.episode_returns):.6f}")
    
    # Test 4: State construction
    print(f"\n=== TEST 4: STATE CONSTRUCTION ===")
    print(f"State components:")
    asset_feat_dim = len(env_short.asset_feature_cols) * env_short.n_assets
    market_feat_dim = len(env_short.market_feature_cols)
    account_feat_dim = 2 * env_short.n_assets if env_short.short_selling_enabled else env_short.n_assets
    
    print(f"  Asset features: {asset_feat_dim} (per-asset technical indicators)")
    print(f"  Market features: {market_feat_dim} (market-wide statistics)")
    print(f"  Account features: {account_feat_dim} (portfolio weights)")
    print(f"  Total state dim: {obs.shape[0]} (should be {asset_feat_dim + market_feat_dim + account_feat_dim})")
    
    print(f"\n🎉 All tests completed!")
    return env_short

if __name__ == "__main__":
    env = test_metatrader_mdp()