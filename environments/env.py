import torch
import numpy as np
import logging
import math

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
    def __init__(self, dataset: dict, feature_columns: list, seq_len: int = 60, min_horizon: int = 45, max_horizon: int = 60, rf_rate=0.02, steps_per_year: int = 252, eta: float = 0.05, eps: float = 1e-12):
        """
        Args:
            dataset: Dict with 'features' and 'raw_prices' tensors
            feature_columns: List of feature names in order
            seq_len: Length of each task sequence
            min_horizon: Minimum episode length within a task
            max_horizon: Maximum episode length within a task
        """
        self.dataset = dataset
        self.feature_columns = feature_columns
        self.seq_len = seq_len
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        # Risk-free handling & DSR hyperparams
        # rf_rate is assumed annual; convert to per-step LOG rf:
        #   rf_step_log = log(1 + rf_rate) / steps_per_year
        # If your rf_rate is already per-step, set steps_per_year=1.
        self.rf_rate = rf_rate
        self.steps_per_year = steps_per_year
        self.rf_step_log = math.log(1.0 + rf_rate) / max(1, steps_per_year)
        self.eta = eta
        self.eps = eps
        
        # Current episode state
        self.current_step = 0
        self.current_task = None
        self.episode_trajectory = []  # Current episode trajectory
        self.terminal_step = None
        self.done = True

        # Tracking (we keep both for interpretability & eval)
        self.log_returns = []         # total portfolio log-returns (incl. cash at rf)
        self.excess_log_returns = []  # excess log-returns over rf (agent's "risk-adjusted" atom)
        # DSR state (EWMA first & second moments of EXCESS returns)
        self.alpha = 0.0
        self.beta = 0.0
        self.capital_history = []

        # Episode tracking
        self.episode_count = 0

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
        
        self.episode_count += 1
        
        # Return initial state: features only [N, F]
        initial_state = self.current_task['features'][0].numpy()  # Convert to numpy
        return initial_state
    
    def step(self, action):
        """
        Take environment step - expects portfolio weights as numpy array
        
        Args:
            action: np.array[N] - portfolio allocation weights (sum ≤ 1)
        """
        if self.done:
            raise ValueError("Episode is done, call reset() first")
            
        # Ensure action is numpy array
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32)
        
        # Current state (normalized features)
        current_state = self.current_task['features'][self.current_step].numpy()  # [N, F]
        
        # Compute reward and normalized allocations
        reward, weights, w_cash = self.compute_reward_with_capital(action)
        
        # Store transition
        self.episode_trajectory.append({
            'state': current_state.copy(),
            'action': action.copy(),
            'reward': reward
        })
        
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
            'capital': self.current_capital,
            'investment_pct': investment_pct,
            'cash_pct': cash_pct,
            'cumulative_return': cumulative_return,
            'pure_return': pure_return,   # kept for backward-compat
            'sharpe_reward': reward,      # name kept for downstream compatibility; now DSR
            'log_return': self.log_returns[-1] if self.log_returns else 0.0,
            'excess_log_return': self.excess_log_returns[-1] if self.excess_log_returns else 0.0,
            'weights': weights.copy(),
            'w_cash': w_cash,
            'step': self.current_step,
            'episode_id': self.episode_count
        }

        
        return next_state, reward, self.done, info
    
    def compute_reward_with_capital(self, portfolio_weights):
        """
        Differential Sharpe Ratio (DSR) reward using portfolio EXCESS LOG-returns.
        - Uses EWMA of first/second moments (alpha, beta) with decay eta.
        - Provides dense, stepwise signal aligned with risk-adjusted performance.
        """
        if self.current_step >= len(self.current_task['raw_prices']) - 1:
            return 0.0  # No next prices available

        # --- 1) Compute asset LOG-returns for the step t -> t+1
        current_prices = self.current_task['raw_prices'][self.current_step].numpy()   # [N]
        next_prices    = self.current_task['raw_prices'][self.current_step + 1].numpy()
        # Guard against zeros/negatives (shouldn't be present but keep robust)
        current_prices = np.clip(current_prices, self.eps, None)
        next_prices    = np.clip(next_prices,    self.eps, None)
        asset_log_returns = np.log(next_prices) - np.log(current_prices)             # [N]

        # --- 2) Portfolio total LOG-return decomposition:
        # total_log_ret = rf + sum_i w_i * (asset_log_i - rf)
        # DSR is defined on EXCESS returns:
        assets_excess = asset_log_returns - self.rf_step_log                         # [N]
        weights, w_cash = normalize_with_budget_constraint(portfolio_weights, self.eps)

        # We do NOT force sum(weights)=1; cash earns rf implicitly via decomposition
        excess_log_return = float(np.dot(weights, assets_excess))                    # scalar
        total_log_return  = self.rf_step_log + excess_log_return                     # scalar

        # --- 3) Update capital using TOTAL log-return (for interpretability)
        self.current_capital *= math.exp(total_log_return)
        self.capital_history.append(self.current_capital)
        self.log_returns.append(total_log_return)
        self.excess_log_returns.append(excess_log_return)

        # --- 4) Differential Sharpe Ratio (DSR) update
        # EWMA state BEFORE update
        alpha_prev = self.alpha
        beta_prev  = self.beta

        # EWMA updates for EXCESS returns
        delta_alpha = self.eta * (excess_log_return - alpha_prev)
        delta_beta  = self.eta * (excess_log_return**2 - beta_prev)
        alpha_new   = alpha_prev + delta_alpha
        beta_new    = beta_prev + delta_beta

        # Variance proxy (ensure positivity)
        var_prev = max(beta_prev - alpha_prev**2, self.eps)
        denom = (var_prev ** 1.5) + self.eps
        # DSR_t using PRE-UPDATE moments (standard form)
        dsr = (beta_prev * delta_alpha - 0.5 * alpha_prev * delta_beta) / denom

        # Commit new EWMA state
        self.alpha = alpha_new
        self.beta  = beta_new

        # Optionally gate very-early steps to avoid noisy spikes
        if self.current_step < 2:
            dsr = 0.0

        # Return both reward and the normalized allocations for logging
        return float(dsr), weights, w_cash

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