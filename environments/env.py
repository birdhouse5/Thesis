import torch
import numpy as np
import logging
import math
from collections import deque

logger = logging.getLogger(__name__)



def normalize_with_budget_constraint(raw_actions: np.ndarray, eps: float = 1e-8):
    """
    Normalize raw action vector into portfolio weights with budget constraint:
    sum(|w_i|) + w_cash = 1, w_cash >= 0
    """
    denom = np.sum(np.abs(raw_actions)) + 1.0 + eps
    weights = raw_actions / denom
    w_cash = 1.0 / denom
    return weights, w_cash


class MetaEnv:
    def __init__(self, dataset: dict, feature_columns: list, seq_len: int = 60, 
                 min_horizon: int = 45, max_horizon: int = 60, rf_rate=0.02, 
                 steps_per_year: int = 252, eta: float = 0.05, eps: float = 1e-12, 
                 transaction_cost_rate: float = 0.001, inflation_rate: float = 0.0,
                 device: str = "cpu"):   # 🔥 NEW
        """
        dataset: Dict with 'features' and 'raw_prices' tensors
        feature_columns: List of feature names
        steps_per_year: 252 for stocks (daily), 35040 for crypto (15-min)
        """
        self.dataset = dataset
        self.feature_columns = feature_columns
        self.seq_len = seq_len
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon

        # DSR / risk params
        self.rf_rate = rf_rate
        self.steps_per_year = steps_per_year
        self.rf_step_log = math.log(1.0 + rf_rate) / max(1, steps_per_year)
        self.eta = eta
        self.eps = eps
        self.transaction_cost_rate = transaction_cost_rate
        self.inflation_rate = inflation_rate

        self.device = torch.device(device)   # 🔥 FIX
        self.prev_weights = None

        # Sequential backtesting
        self.sequential_mode = False
        self.rolling_context = deque(maxlen=self.seq_len)

        # State trackers
        self.current_step = 0
        self.current_task = None
        self.episode_trajectory = []
        self.terminal_step = None
        self.done = True

        self.log_returns = []
        self.excess_log_returns = []
        self.relative_excess_log_returns = []
        self.alpha, self.beta = 0.0, 0.0
        self.capital_history = []
        self.episode_count = 0

        logger.info(f"MetaEnv initialized:")
        logger.info(f"  steps_per_year={self.steps_per_year}, rf_step_log={self.rf_step_log:.8f}")

    def sample_task(self):
        """Sample a random task from the dataset"""
        T = self.dataset['features'].shape[0]
        start = np.random.randint(0, T - self.seq_len)
        
        task_data = {
            'features': self.dataset['features'][start:start + self.seq_len],
            'raw_prices': self.dataset['raw_prices'][start:start + self.seq_len]
        }
        
        return {"sequence": task_data, "task_id": start}

    def set_task(self, task: dict):
        """Set the current task"""
        self.current_task = task["sequence"]  # Now contains both features and raw_prices
        self.task_id = task["task_id"]
        
        # Initialize capital tracking
        self.initial_capital = 100_000.0
        self.current_capital = self.initial_capital
        self.capital_history = [self.initial_capital]
        
        # Reset episode state
        self.current_step = 0
        self.done = False
        self.episode_trajectory = []
        self.terminal_step = np.random.randint(self.min_horizon, self.max_horizon + 1)

    def reset(self):
        """Reset current episode within the current task"""
        if self.current_task is None:
            raise ValueError("Must call set_task() before reset()")
            
        self.current_step = 0
        self.done = False
        self.episode_trajectory = []
        
        # Reset capital tracking for new episode
        self.current_capital = self.initial_capital
        self.capital_history = [self.initial_capital]
        # Reset return series & DSR state
        self.log_returns = []
        self.excess_log_returns = []
        self.alpha = 0.0
        self.beta = 0.0
        self.prev_weights = np.zeros(self.current_task['raw_prices'].shape[1], dtype=np.float32)

        self.episode_count += 1
        
        # Return initial state: features only [N, F]
        initial_state = self.current_task['features'][0].numpy()  # Convert to numpy
        return initial_state
    
    def step(self, action):
        if self.done:
            raise ValueError("Episode is done, call reset() first")

        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32)

        # Normalize → valid portfolio weights
        weights, w_cash = normalize_with_budget_constraint(action)
        
        # Current state (normalized features)
        current_state = self.current_task['features'][self.current_step].numpy()  # [N, F]
        
        # Compute reward and normalized allocations
        reward, weights, w_cash, turnover, cost, equal_weight_log_return, relative_excess_log_return = self.compute_reward_with_capital(action)
        
        # Store transition
        transition_data = {
            'state': current_state.copy(),
            'action': action.copy(),
            'reward': reward
        }

        # Store in appropriate context
        if self.sequential_mode:
            # For sequential backtesting - add to rolling context
            self.rolling_context.append({
                'observations': torch.tensor(current_state, dtype=torch.float32, device=self.device),
                'actions': torch.tensor(weights, dtype=torch.float32, device=self.device),  # Use normalized weights
                'rewards': torch.tensor(reward, dtype=torch.float32, device=self.device)
            })
        else:
            # For episodic training - add to episode trajectory
            self.episode_trajectory.append(transition_data)
        
        # Advance and check termination
        self.current_step += 1
        self.done = (self.current_step >= self.terminal_step) or \
                    (self.current_step >= len(self.current_task['features']))
        
        # Next state
        next_state = (np.zeros_like(current_state) if self.done 
                     else self.current_task['features'][self.current_step].numpy())
        
        # Performance metrics (based on normalized allocations)
        investment_pct = np.sum(np.abs(weights))
        cash_pct = w_cash
        cumulative_return = (self.current_capital - self.initial_capital) / self.initial_capital
        pure_return = self.log_returns[-1] if self.log_returns else 0.0

        info = {
            # === Portfolio Composition ===
            'capital': self.current_capital,
            'investment_pct': investment_pct,
            'cash_pct': w_cash,
            'cumulative_return': cumulative_return,
            "weights": weights.copy(),                    # Portfolio allocation
            'weights_long': weights[weights > 0].sum(),   # Long exposure
            'weights_short': abs(weights[weights < 0]).sum(), # Short exposure
            'net_exposure': np.sum(weights),             
            'gross_exposure': np.sum(np.abs(weights)),
            'portfolio_concentration': np.sum(weights**2), # HHI concentration
            'num_active_positions': np.sum(np.abs(weights) > 0.01), # Active positions
            
            # === Returns & Performance ===
            'log_return': self.log_returns[-1] if self.log_returns else 0.0,
            'excess_log_return': self.excess_log_returns[-1] if self.excess_log_returns else 0.0,
            'pure_return': pure_return,
            'sharpe_reward': reward,  # DSR reward
            
            # === DSR State Tracking ===
            'dsr_alpha': self.alpha,                     # EWMA first moment
            'dsr_beta': self.beta,                       # EWMA second moment
            'dsr_variance': max(self.beta - self.alpha**2, self.eps), # Current variance estimate
            
            # === Transaction Costs ===
            'transaction_cost': cost,
            'turnover': turnover,
            
            # === Environment Context ===
            'task_id': getattr(self, 'task_id', -1),
            'terminal_step': self.terminal_step,
            'step': self.current_step,
            'episode_id': self.episode_count,
            
            # === Market Context ===
            'current_prices': current_prices.copy() if 'current_prices' in locals() else np.array([]),
            'price_changes': (next_prices - current_prices) if 'next_prices' in locals() else np.array([]),
            'market_return': np.mean(asset_log_returns) if 'asset_log_returns' in locals() else 0.0,
            'benchmark_log_return': equal_weight_log_return,
            'relative_excess_log_return': relative_excess_log_return,
        }

        
        return next_state, reward, self.done, info
    
    def compute_reward_with_capital(self, portfolio_weights):
        """
        Differential Sharpe Ratio (DSR) reward using portfolio EXCESS LOG-returns.
        - Uses EWMA of first/second moments (alpha, beta) with decay eta.
        - Provides dense, stepwise signal aligned with risk-adjusted performance.
        
        Returns:
            tuple: (reward, weights, w_cash, turnover, cost)
        """
        if self.current_step >= len(self.current_task['raw_prices']) - 1:
            weights = np.zeros_like(portfolio_weights, dtype=np.float32)
            w_cash = 1.0
            return 0.0, weights, w_cash, 0.0, 0.0, 0.0, 0.0

        # --- 1) Compute asset LOG-returns for the step t -> t+1
        current_prices = self.current_task['raw_prices'][self.current_step].numpy()   # [N]
        next_prices    = self.current_task['raw_prices'][self.current_step + 1].numpy()
        current_prices = np.clip(current_prices, self.eps, None)
        next_prices    = np.clip(next_prices,    self.eps, None)
        asset_log_returns = np.log(next_prices) - np.log(current_prices)  # [N]
        equal_weight_log_return = np.mean(asset_log_returns)

        # --- 2) Normalize portfolio weights FIRST
        weights, w_cash = normalize_with_budget_constraint(portfolio_weights, self.eps)

        # Transaction costs (proportional to turnover)
        if self.prev_weights is None:
            turnover = np.sum(np.abs(weights))  # first step baseline
        else:
            turnover = np.sum(np.abs(weights - self.prev_weights))
        cost = self.transaction_cost_rate * turnover

        # --- 3) Portfolio total LOG-return decomposition
        assets_excess = asset_log_returns - self.rf_step_log
        equal_weight_excess = equal_weight_log_return - self.rf_step_log
        agent_excess = np.dot(weights, assets_excess) - cost

        # Excess log return of portfolio
        excess_log_return = float(np.dot(weights, assets_excess)) - cost
        total_log_return  = self.rf_step_log + excess_log_return
        relative_excess_log_return = agent_excess - equal_weight_excess

        self.prev_weights = weights

        # --- 4) Update capital & history
        self.current_capital *= math.exp(total_log_return)
        self.capital_history.append(self.current_capital)
        self.log_returns.append(total_log_return)
        self.excess_log_returns.append(excess_log_return)
        self.relative_excess_log_returns.append(relative_excess_log_return)

        # --- 5) Differential Sharpe Ratio (DSR) update
        alpha_prev = self.alpha
        beta_prev  = self.beta
        delta_alpha = self.eta * (relative_excess_log_return - alpha_prev)
        delta_beta  = self.eta * (relative_excess_log_return**2 - beta_prev)
        alpha_new   = alpha_prev + delta_alpha
        beta_new    = beta_prev + delta_beta

        var_prev = max(beta_prev - alpha_prev**2, self.eps)
        denom = (var_prev ** 1.5) + self.eps
        dsr = (beta_prev * delta_alpha - 0.5 * alpha_prev * delta_beta) / denom
        dsr -= self.inflation_rate * w_cash

        self.alpha = alpha_new
        self.beta  = beta_new

        if self.current_step < 2:  # gate very-early steps
            dsr = 0.0

        return float(dsr), weights, w_cash, turnover, cost, equal_weight_log_return, relative_excess_log_return


    def rollout_episode(self, policy, encoder):
        """
        Perform a complete episode rollout using policy and encoder
        
        Args:
            policy: function that takes (state, latent) -> action
            encoder: encoder that takes trajectory context -> latent distribution
            
        Returns:
            trajectory: List of dicts with keys ['state', 'action', 'reward', 'latent']
        """
        if self.current_task is None:
            raise ValueError("Must call set_task() before rollout")
            
        trajectory = []
        state = self.reset()
        
        # Keep track of trajectory context for encoder (following paper's τ:t notation)
        trajectory_context = []  # This will build up the τ:t sequence
        
        step = 0
        while not self.done:
            # Build context τ:t for encoder (states, actions, rewards up to current time)
            if step == 0:
                # At t=0, we only have initial state s0
                context_for_encoder = trajectory_context  # Empty list
            else:
                # At t>0, we have transitions up to current state
                context_for_encoder = trajectory_context
            
            # Encode current trajectory context to get latent
            latent_dist = encoder.encode(context_for_encoder)
            latent = latent_dist.sample() if hasattr(latent_dist, 'sample') else latent_dist
            
            # Policy computes action given current state and latent
            action = policy(state, latent)
            
            # Take environment step
            next_state, reward, done, info = self.step(action)
            
            # Store full transition info for training
            trajectory.append({
                'state': state.copy(),           # [N, F]
                'action': info['weights'].copy(),   # normalized portfolio weights
                'reward': reward,                # scalar
                'latent': latent.copy() if hasattr(latent, 'copy') else latent,  # [latent_dim]
                'next_state': next_state.copy() if not done else None
            })

            trajectory_context.append({
                'state': state.copy(),
                'action': info['weights'].copy(),   # normalized portfolio weights
                'reward': reward
            })
            
            state = next_state
            step += 1
            
        return trajectory

    def get_trajectory_for_vae_training(self):
        """
        Get the current episode trajectory in format suitable for VAE training
        
        Returns:
            trajectory: List of dicts suitable for encoder training
        """
        return self.episode_trajectory.copy()

    def get_task_info(self):
        """Get information about current task"""
        return {
            'task_id': getattr(self, 'task_id', None),
            'current_step': self.current_step,
            'terminal_step': self.terminal_step,
            'done': self.done
        }

    def get_rolling_context_size(self):
            """Get current size of rolling context buffer."""
            return len(self.rolling_context) if self.sequential_mode else 0


    def set_sequential_mode(self, enabled: bool = True):
        """
        Enable/disable sequential mode for backtesting.
        In sequential mode, context persists across steps for rolling window backtesting.
        """
        self.sequential_mode = enabled
        if not enabled:
            self.rolling_context.clear()
        logger.info(f"Sequential mode {'enabled' if enabled else 'disabled'}")