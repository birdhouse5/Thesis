import torch
import numpy as np
import logging
import math
from collections import deque

logger = logging.getLogger(__name__)



def normalize_with_budget_constraint(raw_actions, eps: float = 1e-8):
    """
    Normalize raw action vector into portfolio weights with budget constraint:
    sum(|w_i|) + w_cash = 1, w_cash >= 0

    Accepts either torch.Tensor or numpy.ndarray and returns the same type.
    """
    is_torch = torch.is_tensor(raw_actions)
    actions_t = raw_actions if is_torch else torch.as_tensor(raw_actions, dtype=torch.float32)

    denom = torch.sum(torch.abs(actions_t)) + 1.0 + eps
    weights_t = actions_t / denom
    w_cash_t = 1.0 / denom

    if is_torch:
        return weights_t, w_cash_t
    else:
        return weights_t.detach().cpu().numpy(), float(w_cash_t.detach().cpu().item())


class MetaEnv:
    def __init__(self, dataset: dict, feature_columns: list, seq_len: int = 60, 
                 min_horizon: int = 45, max_horizon: int = 60, rf_rate=0.02, 
                 steps_per_year: int = 252, eta: float = 0.05, eps: float = 1e-12, 
                 transaction_cost_rate: float = 0.001, inflation_rate: float = 0.0,
                 reward_type: str = "dsr", reward_lookback: int = 20,
                 device: str = "cpu", eval_mode: bool = False):
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
        self.portfolio_values = []  # For drawdown calculation
        self.recent_returns = []    # For Sharpe calculation

        self.eval_mode = eval_mode

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
        self.prev_weights = torch.zeros(int(self.current_task['raw_prices'].shape[1]), dtype=torch.float32, device=self.device)

        self.episode_count += 1
        
        # Return initial state: features only [N, F] as torch tensor
        initial_state = self.current_task['features'][0].to(self.device).to(torch.float32)
        return initial_state
    
    def step(self, action):
        if self.done:
            raise ValueError("Episode is done, call reset() first")

        # Ensure torch tensor on device
        if not torch.is_tensor(action):
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        else:
            action = action.to(self.device, dtype=torch.float32)

        # Normalize → valid portfolio weights (once)
        weights, w_cash = normalize_with_budget_constraint(action)
        w_cash_scalar = float(w_cash.item()) if torch.is_tensor(w_cash) else float(w_cash)
        
        # Current state (normalized features) as torch tensor
        current_state = self.current_task['features'][self.current_step].to(self.device).to(torch.float32)  # [N, F]
        
        # Compute reward using pre-normalized weights
        reward, weights, w_cash_scalar, turnover, cost, equal_weight_log_return, relative_excess_log_return = self.compute_reward_with_capital(weights)
        
        # Store transition
        transition_data = {
            'state': current_state.clone(),
            'action': weights.clone(),
            'reward': reward
        }

        # Store in appropriate context
        if self.sequential_mode:
            # For sequential backtesting - add to rolling context
            self.rolling_context.append({
                'observations': current_state.detach(),
                'actions': weights.detach(),
                'rewards': torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            })
        else:
            # For episodic training - add to episode trajectory
            self.episode_trajectory.append(transition_data)
        
        # Advance and check termination
        self.current_step += 1
        self.done = (self.current_step >= self.terminal_step) or \
                    (self.current_step >= len(self.current_task['features']))
        
        # Next state
        if self.done:
            next_state = torch.zeros_like(current_state)
        else:
            next_state = self.current_task['features'][self.current_step].to(self.device).to(torch.float32)
        
        # Performance metrics (based on normalized allocations)
        investment_pct = float(torch.sum(torch.abs(weights)).item())
        cash_pct = w_cash_scalar
        cumulative_return = (self.current_capital - self.initial_capital) / self.initial_capital
        pure_return = self.log_returns[-1] if self.log_returns else 0.0

        # Base info (minimal)
        info = {
            'capital': self.current_capital,
            'investment_pct': investment_pct,
            'cash_pct': cash_pct,
            'cumulative_return': cumulative_return,
            'weights': weights.detach(),
            'log_return': self.log_returns[-1] if self.log_returns else 0.0,
            'excess_log_return': self.excess_log_returns[-1] if self.excess_log_returns else 0.0,
            'pure_return': pure_return,
            'sharpe_reward': reward,
            'task_id': getattr(self, 'task_id', -1),
            'terminal_step': self.terminal_step,
            'step': self.current_step,
            'episode_id': self.episode_count,
        }

        # Extra metrics only if eval_mode
        if self.eval_mode:
            info.update({
                'weights_long': float(weights[weights > 0].sum().item()),
                'weights_short': float(torch.abs(weights[weights < 0]).sum().item()),
                'net_exposure': float(torch.sum(weights).item()),
                'gross_exposure': float(torch.sum(torch.abs(weights)).item()),
                'portfolio_concentration': float(torch.sum(weights ** 2).item()),
                'num_active_positions': int((torch.abs(weights) > 0.01).sum().item()),
                'transaction_cost': cost,
                'turnover': turnover,
                'benchmark_log_return': equal_weight_log_return,
                'relative_excess_log_return': relative_excess_log_return,
            })

        
        return next_state, reward, self.done, info
    
    def compute_reward_with_capital(self, portfolio_weights):
        """Compute reward based on configured reward type."""
        if self.reward_type == "dsr":
            return self._compute_dsr_reward(portfolio_weights)
        elif self.reward_type == "sharpe":
            return self._compute_sharpe_reward(portfolio_weights)
        elif self.reward_type == "drawdown":
            return self._compute_drawdown_reward(portfolio_weights)
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

    def _compute_dsr_reward(self, portfolio_weights):
        """
        Differential Sharpe Ratio (DSR) reward using portfolio EXCESS LOG-returns.
        - Uses EWMA of first/second moments (alpha, beta) with decay eta.
        - Provides dense, stepwise signal aligned with risk-adjusted performance.
        
        Returns:
            tuple: (reward, weights, w_cash, turnover, cost)
        """
        if self.current_step >= len(self.current_task['raw_prices']) - 1:
            if torch.is_tensor(portfolio_weights):
                weights = torch.zeros_like(portfolio_weights, dtype=torch.float32, device=self.device)
            else:
                weights = torch.zeros(len(portfolio_weights), dtype=torch.float32, device=self.device)
            w_cash = 1.0
            return 0.0, weights, float(w_cash), 0.0, 0.0, 0.0, 0.0

        # --- 1) Compute asset LOG-returns for the step t -> t+1
        current_prices = self.current_task['raw_prices'][self.current_step].to(self.device).to(torch.float32)
        next_prices    = self.current_task['raw_prices'][self.current_step + 1].to(self.device).to(torch.float32)
        current_prices = torch.clamp(current_prices, min=self.eps)
        next_prices    = torch.clamp(next_prices,    min=self.eps)
        asset_log_returns = torch.log(next_prices) - torch.log(current_prices)  # [N]
        equal_weight_log_return = torch.mean(asset_log_returns)

        # --- 2) Use provided weights if already normalized; otherwise normalize
        if torch.is_tensor(portfolio_weights):
            weights = portfolio_weights.to(self.device, dtype=torch.float32)
            # infer cash from budget constraint
            gross = torch.sum(torch.abs(weights))
            w_cash_scalar_t = 1.0 / (gross + 1.0 + self.eps)
        else:
            weights, w_cash = normalize_with_budget_constraint(portfolio_weights, self.eps)
            weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
            w_cash_scalar_t = w_cash if torch.is_tensor(w_cash) else torch.tensor(w_cash, dtype=torch.float32, device=self.device)

        # Transaction costs (proportional to turnover)
        if self.prev_weights is None:
            turnover_t = torch.sum(torch.abs(weights))
        else:
            turnover_t = torch.sum(torch.abs(weights - self.prev_weights))
        cost_t = torch.as_tensor(self.transaction_cost_rate, dtype=torch.float32, device=self.device) * turnover_t

        # --- 3) Portfolio total LOG-return decomposition
        assets_excess = asset_log_returns - self.rf_step_log
        equal_weight_excess = equal_weight_log_return - self.rf_step_log
        agent_excess_t = torch.dot(weights, assets_excess) - cost_t

        # Excess log return of portfolio
        excess_log_return_t = torch.dot(weights, assets_excess) - cost_t
        total_log_return_t  = torch.as_tensor(self.rf_step_log, dtype=torch.float32, device=self.device) + excess_log_return_t
        relative_excess_log_return_t = agent_excess_t - equal_weight_excess

        self.prev_weights = weights

        # --- 4) Update capital & history
        self.current_capital *= float(torch.exp(total_log_return_t).item())
        self.capital_history.append(self.current_capital)
        self.log_returns.append(float(total_log_return_t.item()))
        self.excess_log_returns.append(float(excess_log_return_t.item()))
        self.relative_excess_log_returns.append(float(relative_excess_log_return_t.item()))

        # --- 5) Differential Sharpe Ratio (DSR) update
        alpha_prev = self.alpha
        beta_prev  = self.beta
        delta_alpha = self.eta * (relative_excess_log_return_t - alpha_prev)
        delta_beta  = self.eta * (relative_excess_log_return_t**2 - beta_prev)
        alpha_new   = alpha_prev + delta_alpha
        beta_new    = beta_prev + delta_beta

        var_prev = max(beta_prev - alpha_prev**2, self.eps)
        denom = (var_prev ** 1.5) + self.eps
        dsr_t = (beta_prev * delta_alpha - 0.5 * alpha_prev * delta_beta) / denom
        dsr_t = dsr_t - self.inflation_rate * w_cash_scalar_t

        self.alpha = alpha_new
        self.beta  = beta_new

        if self.current_step < 2:  # gate very-early steps
            dsr_t = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        
        return float(dsr_t.item()), weights, float(w_cash_scalar_t.item()), float(turnover_t.item()), float(cost_t.item()), float(equal_weight_log_return.item()), float(relative_excess_log_return_t.item())
    
    def _compute_sharpe_reward(self, portfolio_weights):
        """Simple rolling Sharpe ratio reward."""
        # === Input validation ===
        if self.current_step >= len(self.current_task['raw_prices']) - 1:
            if torch.is_tensor(portfolio_weights):
                weights = torch.zeros_like(portfolio_weights, dtype=torch.float32, device=self.device)
            else:
                weights = torch.zeros(len(portfolio_weights), dtype=torch.float32, device=self.device)
            w_cash = 1.0
            return 0.0, weights, float(w_cash), 0.0, 0.0, 0.0, 0.0

        # === Asset returns ===
        current_prices = self.current_task['raw_prices'][self.current_step].to(self.device).to(torch.float32)
        next_prices = self.current_task['raw_prices'][self.current_step + 1].to(self.device).to(torch.float32)
        current_prices = torch.clamp(current_prices, min=self.eps)
        next_prices = torch.clamp(next_prices, min=self.eps)
        asset_log_returns = torch.log(next_prices) - torch.log(current_prices)
        equal_weight_log_return = torch.mean(asset_log_returns)

        # === Weight normalization ===
        if torch.is_tensor(portfolio_weights):
            weights = portfolio_weights.to(self.device, dtype=torch.float32)
            gross = torch.sum(torch.abs(weights))
            w_cash_scalar_t = 1.0 / (gross + 1.0 + self.eps)
        else:
            weights, w_cash = normalize_with_budget_constraint(portfolio_weights, self.eps)
            weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
            w_cash_scalar_t = w_cash if torch.is_tensor(w_cash) else torch.tensor(w_cash, dtype=torch.float32, device=self.device)

        # === Transaction costs ===
        if self.prev_weights is None:
            turnover_t = torch.sum(torch.abs(weights))
        else:
            turnover_t = torch.sum(torch.abs(weights - self.prev_weights))
        cost_t = torch.as_tensor(self.transaction_cost_rate, dtype=torch.float32, device=self.device) * turnover_t

        # === Portfolio returns ===
        assets_excess = asset_log_returns - self.rf_step_log
        equal_weight_excess = equal_weight_log_return - self.rf_step_log
        agent_excess_t = torch.dot(weights, assets_excess) - cost_t
        excess_log_return_t = torch.dot(weights, assets_excess) - cost_t
        total_log_return_t = torch.as_tensor(self.rf_step_log, dtype=torch.float32, device=self.device) + excess_log_return_t
        relative_excess_log_return_t = agent_excess_t - equal_weight_excess
        self.prev_weights = weights

        # === Update tracking ===
        self.current_capital *= float(torch.exp(total_log_return_t).item())
        self.capital_history.append(self.current_capital)
        self.log_returns.append(float(total_log_return_t.item()))
        self.excess_log_returns.append(float(excess_log_return_t.item()))
        self.relative_excess_log_returns.append(float(relative_excess_log_return_t.item()))

        # === Sharpe-specific reward calculation ===
        self.recent_returns.append(float(excess_log_return_t.item()))
        if len(self.recent_returns) > self.reward_lookback:
            self.recent_returns.pop(0)

        if len(self.recent_returns) >= 10:
            returns_array = np.array(self.recent_returns)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            sharpe_reward = mean_return / (std_return + 1e-8)
        else:
            sharpe_reward = 0.0

        return float(sharpe_reward), weights, float(w_cash_scalar_t.item() if torch.is_tensor(w_cash_scalar_t) else w_cash_scalar_t), float(turnover_t.item()), float(cost_t.item()), float(equal_weight_log_return.item()), float(relative_excess_log_return_t.item())


    def _compute_drawdown_reward(self, portfolio_weights):
        """Maximum drawdown-based reward."""
        # === Input validation ===
        if self.current_step >= len(self.current_task['raw_prices']) - 1:
            if torch.is_tensor(portfolio_weights):
                weights = torch.zeros_like(portfolio_weights, dtype=torch.float32, device=self.device)
            else:
                weights = torch.zeros(len(portfolio_weights), dtype=torch.float32, device=self.device)
            w_cash = 1.0
            return 0.0, weights, float(w_cash), 0.0, 0.0, 0.0, 0.0

        # === Asset returns ===
        current_prices = self.current_task['raw_prices'][self.current_step].to(self.device).to(torch.float32)
        next_prices = self.current_task['raw_prices'][self.current_step + 1].to(self.device).to(torch.float32)
        current_prices = torch.clamp(current_prices, min=self.eps)
        next_prices = torch.clamp(next_prices, min=self.eps)
        asset_log_returns = torch.log(next_prices) - torch.log(current_prices)
        equal_weight_log_return = torch.mean(asset_log_returns)

        # === Weight normalization ===
        if torch.is_tensor(portfolio_weights):
            weights = portfolio_weights.to(self.device, dtype=torch.float32)
            gross = torch.sum(torch.abs(weights))
            w_cash_scalar_t = 1.0 / (gross + 1.0 + self.eps)
        else:
            weights, w_cash = normalize_with_budget_constraint(portfolio_weights, self.eps)
            weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
            w_cash_scalar_t = w_cash if torch.is_tensor(w_cash) else torch.tensor(w_cash, dtype=torch.float32, device=self.device)

        # === Transaction costs ===
        if self.prev_weights is None:
            turnover_t = torch.sum(torch.abs(weights))
        else:
            turnover_t = torch.sum(torch.abs(weights - self.prev_weights))
        cost_t = torch.as_tensor(self.transaction_cost_rate, dtype=torch.float32, device=self.device) * turnover_t

        # === Portfolio returns ===
        assets_excess = asset_log_returns - self.rf_step_log
        equal_weight_excess = equal_weight_log_return - self.rf_step_log
        agent_excess_t = torch.dot(weights, assets_excess) - cost_t
        excess_log_return_t = torch.dot(weights, assets_excess) - cost_t
        total_log_return_t = torch.as_tensor(self.rf_step_log, dtype=torch.float32, device=self.device) + excess_log_return_t
        relative_excess_log_return_t = agent_excess_t - equal_weight_excess
        self.prev_weights = weights

        # === Update capital and tracking ===
        self.current_capital *= float(torch.exp(total_log_return_t).item())
        self.capital_history.append(self.current_capital)
        self.log_returns.append(float(total_log_return_t.item()))
        self.excess_log_returns.append(float(excess_log_return_t.item()))
        self.relative_excess_log_returns.append(float(relative_excess_log_return_t.item()))

        # === Drawdown-specific reward calculation ===
        self.portfolio_values.append(self.current_capital)
        if len(self.portfolio_values) > self.reward_lookback:
            self.portfolio_values.pop(0)

        if len(self.portfolio_values) >= 10:
            values_array = np.array(self.portfolio_values)
            peak = np.maximum.accumulate(values_array)
            drawdown = (values_array - peak) / peak
            max_drawdown = np.min(drawdown)
            # Reward is negative of max drawdown (less negative is better)
            drawdown_reward = -max_drawdown
        else:
            drawdown_reward = 0.0

        return float(drawdown_reward), weights, float(w_cash_scalar_t.item() if torch.is_tensor(w_cash_scalar_t) else w_cash_scalar_t), float(turnover_t.item()), float(cost_t.item()), float(equal_weight_log_return.item()), float(relative_excess_log_return_t.item())


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