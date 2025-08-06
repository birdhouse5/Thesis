import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MetaEnv:
    def __init__(self, dataset: torch.Tensor, seq_len: int = 60, min_horizon: int = 45, max_horizon: int = 60):
        """
        Args:
            dataset: Tensor[T x N x F] - full time series data
            seq_len: length of each task sequence
            min_horizon: minimum episode length within a task
            max_horizon: maximum episode length within a task
        """
        self.dataset = dataset
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

        # Episode tracking
        self.episode_count = 0

    def sample_task(self):
        """Sample a random task from the dataset"""
        T = self.dataset.shape[0]
        start = torch.randint(0, T - self.seq_len, (1,)).item()
        task = self.dataset[start:start + self.seq_len]
        return {"sequence": task, "task_id": start}

    def set_task(self, task: dict):
        """Set the current task - this defines the MDP for this meta-episode"""
        self.current_task = task["sequence"]  # Shape: [seq_len, N, F]
        self.task_id = task["task_id"]
        
        # Reset episode state
        self.current_step = 0
        self.done = False
        self.episode_trajectory = []
        
        # Sample episode length for this task
        self.terminal_step = torch.randint(self.min_horizon, self.max_horizon + 1, (1,)).item()

    def reset(self):
        """Reset current episode within the current task"""
        if self.current_task is None:
            raise ValueError("Must call set_task() before reset()")
            
        self.current_step = 0
        self.done = False
        self.episode_trajectory = []

        self.returns = []
        
        self.episode_count += 1
        
        # Return initial state: shape [N, F] (no batch dimension during rollout)
        initial_state = self.current_task[0]  # Shape: [N, F]
        return initial_state

    def step(self, action: torch.Tensor):
        """
        Take environment step
        
        Args:
            action: Tensor[N] - portfolio allocation
            
        Returns:
            next_state: Tensor[N, F] - next observation
            reward: float - scalar reward
            done: bool - episode termination
            info: dict - additional info
        """
        if self.done:
            raise ValueError("Episode is done, call reset() first")
            
        # Current state
        current_state = self.current_task[self.current_step]  # Shape: [N, F]
        
        # Compute reward based on current state and action
        reward = self.compute_reward(current_state, action)
        
        # Store transition in episode trajectory
        # Note: storing state without batch dimension for encoder compatibility
        self.episode_trajectory.append({
            'state': current_state.clone(),  # [N, F]
            'action': action.clone(),        # [N] or dict
            'reward': reward.item()          # scalar
        })
        
        # Advance step
        self.current_step += 1
        
        # Check if episode is done
        self.done = (self.current_step >= self.terminal_step) or (self.current_step >= len(self.current_task))
        
        # Get next state
        if self.done:
            # Terminal state - could be zeros or repeat last state
            next_state = torch.zeros_like(current_state)
        else:
            next_state = self.current_task[self.current_step]
            
        # Log DSR components
        logger.debug(f"Step {self.current_step}: reward={reward:.6f}, "
                     f"returns_count={len(self.returns)}")
            
        return next_state, reward.item(), self.done, {
            'task_id': self.task_id,
            'portfolio_weights': action if not isinstance(action, dict) else None,
            'step': self.current_step,
            'episode_id': self.episode_count
        }

    def compute_reward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Compute reward based on Sharpe ratio
        
        Args:
            state: Tensor[N, F] - current market state
            action: Tensor[N] - portfolio allocation weights
            
        Returns:
            reward: Tensor (scalar)
        """
        # Convert action dict if needed (from hierarchical policy)
        if isinstance(action, dict):
            portfolio_weights = self._discretize_action(action)
            portfolio_weights = torch.from_numpy(portfolio_weights).float()
        else:
            portfolio_weights = action
            
        return self.sharpe_ratio(state, portfolio_weights)

    def sharpe_ratio(self, state: torch.Tensor, portfolio_weights: torch.Tensor):
        """
        Compute Sharpe ratio reward
        
        Args:
            state: Tensor[N, F] - current market state
            portfolio_weights: Tensor[N] - portfolio allocation
            
        Returns:
            reward: Tensor (scalar)
        """
        # Calculate portfolio return
        R_t = self._calculate_portfolio_return(state, portfolio_weights)
        self.returns.append(R_t.item())

        # Need at least 2 returns to calculate Sharpe
        if len(self.returns) < 2:
            return torch.tensor(0.0)
        
        # Calculate Sharpe ratio
        returns_array = torch.tensor(self.returns)
        mean_return = returns_array.mean()
        std_return = returns_array.std()
        
        # Avoid division by zero
        if std_return < 1e-8:
            sharpe = torch.tensor(0.0)
        else:
            sharpe = mean_return / std_return
        
        return sharpe

    def _calculate_portfolio_return(self, current_state: torch.Tensor, portfolio_weights: torch.Tensor):
        """
        Calculate portfolio return based on current and next states
        
        Args:
            current_state: Tensor[N, F] - current market state
            portfolio_weights: Tensor[N] - portfolio allocation
            
        Returns:
            portfolio_return: Tensor (scalar)
        """
        if self.current_step >= len(self.current_task) - 1:
            return torch.tensor(0.0)  # No next observation available
        
        next_state = self.current_task[self.current_step + 1]  # [N, F]
        
        # Get close price index (assume it's the first feature or find it)
        close_idx = self._get_close_price_idx()
        
        current_prices = current_state[:, close_idx]  # [N]
        next_prices = next_state[:, close_idx]        # [N]
        
        # Calculate individual asset returns: (price_t+1 - price_t) / price_t
        asset_returns = (next_prices - current_prices) / (current_prices + 1e-8)  # Avoid div by 0
        
        # Calculate portfolio return as weighted sum
        portfolio_return = torch.sum(portfolio_weights * asset_returns)
        
        return portfolio_return

    def _get_close_price_idx(self):
        """Get the index of close price in feature columns"""
        # This assumes 'close_norm' is in your selected features
        # You'll need to adjust based on your actual feature selection
        # For now, assume first feature is close price
        return 0

    def _discretize_action(self, action_dict):
        """Convert hierarchical action to portfolio weights (numpy version)"""
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
        N = len(self.current_task[0])  # Number of assets
        portfolio_weights = np.zeros(N)
        
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
        
        return portfolio_weights

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