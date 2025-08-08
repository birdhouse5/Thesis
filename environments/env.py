import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MetaEnv:
    def __init__(self, dataset: dict, feature_columns: list, seq_len: int = 60, min_horizon: int = 45, max_horizon: int = 60):
        """
        Args:
            dataset: Dict with 'features' and 'raw_prices' tensors
            feature_comuns: List of feature names in order
            seq_len: length of each task sequence
            min_horizon: minimum episode length within a task
            max_horizon: maximum episode length within a task
        """
        self.dataset = dataset
        self.feature_columns = feature_columns
        self.seq_len = seq_len
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        
        # Current episode state
        self.current_step = 0
        self.current_task = None
        self.episode_trajectory = []  # Current episode trajectory
        self.terminal_step = None
        self.done = True

        self.returns = []
        self.capital_history = []

        # Episode tracking
        self.episode_count = 0
        
        self._close_idx_cache = self._find_close_price_idx()

    def _find_close_price_idx(self):
        """Find close_norm index in feature columns"""
        try:
            return self.feature_columns.index('close_norm')
        except ValueError:
            # Fallback options
            for idx, col in enumerate(self.feature_columns):
                if 'close' in col.lower() and 'norm' in col.lower():
                    return idx
            raise ValueError("Could not find close price feature (close_norm) in feature columns")

    def sample_task(self):
        """Sample a random task from the dataset"""
        T = self.dataset['features'].shape[0]
        start = torch.randint(0, T - self.seq_len, (1,)).item()
        
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
        self.terminal_step = torch.randint(self.min_horizon, self.max_horizon + 1, (1,)).item() 

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
        self.returns = []  # Reset returns for fresh Sharpe calculation
        
        self.episode_count += 1
        
        # Return initial state: features only [N, F]
        initial_state = self.current_task['features'][0]
        return initial_state
    
    def step(self, action: torch.Tensor):
        """
        Take environment step - now expects direct portfolio weights
        
        Args:
            action: Tensor[N] - portfolio allocation weights (sum ≤ 1)
        """
        if self.done:
            raise ValueError("Episode is done, call reset() first")
            
        # Current state (normalized features)
        current_state = self.current_task['features'][self.current_step]  # [N, F]
        
        # Compute Sharpe ratio reward (updates capital tracking)
        reward = self.compute_reward_with_capital(action)
        
        # Store transition
        self.episode_trajectory.append({
            'state': current_state.clone(),
            'action': action.clone(),        # Direct portfolio weights
            'reward': reward.item()
        })
        
        # Advance and check termination
        self.current_step += 1
        self.done = (self.current_step >= self.terminal_step) or \
                (self.current_step >= len(self.current_task['features']))
        
        # Next state
        next_state = (torch.zeros_like(current_state) if self.done 
                    else self.current_task['features'][self.current_step])
        
        # Performance metrics
        investment_pct = action.sum().item()
        cash_pct = 1.0 - investment_pct
        cumulative_return = (self.current_capital - self.initial_capital) / self.initial_capital
        pure_return = self.returns[-1] if self.returns else 0.0
        
        info = {
            'capital': self.current_capital,
            'investment_pct': investment_pct,
            'cash_pct': cash_pct, 
            'cumulative_return': cumulative_return,
            'pure_return': pure_return,
            'sharpe_reward': reward.item(),
            'step': self.current_step,
            'episode_id': self.episode_count
        }
        
        return next_state, reward.item(), self.done, info
    

    def compute_reward_with_capital(self, portfolio_weights: torch.Tensor):
        """
        Compute Sharpe ratio reward while tracking capital performance
        
        Args:
            portfolio_weights: Tensor[N] - portfolio allocation weights
            
        Returns:
            reward: Tensor (scalar) - Sharpe ratio for RL optimization
        """
        if self.current_step >= len(self.current_task['raw_prices']) - 1:
            return torch.tensor(0.0)  # No next prices available
        
        # Get current and next raw prices
        current_prices = torch.tensor(self.current_task['raw_prices'][self.current_step])     # [N]
        next_prices = torch.tensor(self.current_task['raw_prices'][self.current_step + 1])   # [N]
        
        # Calculate individual asset returns
        asset_returns = (next_prices - current_prices) / current_prices  # [N]
        
        # Calculate portfolio return: weighted sum of asset returns
        portfolio_return = torch.sum(portfolio_weights * asset_returns)
        
        # Update capital tracking
        old_capital = self.current_capital
        self.current_capital = old_capital * (1.0 + portfolio_return.item())
        self.capital_history.append(self.current_capital)
        
        # Store return for Sharpe calculation
        self.returns.append(portfolio_return.item())
        
        # Calculate Sharpe ratio as RL reward
        if len(self.returns) >= 2:
            returns_tensor = torch.tensor(self.returns)
            mean_return = returns_tensor.mean()
            std_return = returns_tensor.std()
            sharpe = mean_return / (std_return + 1e-8)
        else:
            sharpe = torch.tensor(0.0)
        
        return sharpe


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
                'state': state.clone(),      # [N, F]
                'action': action.clone(),    # [N] 
                'reward': reward,            # scalar
                'latent': latent.clone(),    # [latent_dim]
                'next_state': next_state.clone() if not done else None
            })
            
            # Update trajectory context for next iteration
            # This follows paper's τ:t notation: (s0,a0,r1,s1,a1,r2,...,st)
            trajectory_context.append({
                'state': state.clone(),
                'action': action.clone(),
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

# Example usage:
if __name__ == "__main__":
    # Create dummy dataset
    T, N, F = 1000, 10, 5  # 1000 timesteps, 10 assets, 5 features
    dataset = torch.randn(T, N, F)
    
    # Create environment
    env = MetaEnv(dataset, seq_len=60)
    
    # Sample and set task
    task = env.sample_task()
    env.set_task(task)
    
    print(f"Task sequence shape: {task['sequence'].shape}")
    print(f"Task ID: {task['task_id']}")
    
    # Reset and take a few steps
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    for i in range(5):
        action = torch.randn(N) * 0.1  # Random portfolio allocation
        next_state, reward, done, info = env.step(action)
        print(f"Step {i}: reward={reward:.4f}, done={done}")
        if done:
            break
        state = next_state