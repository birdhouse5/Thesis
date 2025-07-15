# varibad_portfolio_env.py
"""
VariBAD implementation for portfolio optimization.
Phase 1: Basic implementation with 30-day tasks and market regime learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import gym
from gym import spaces

from sp500_loader.core.environment import PortfolioEnv, DifferentialSharpeRatio


class PortfolioVariationalEncoder(nn.Module):
    """
    Variational encoder for portfolio task inference.
    Encodes trajectory (states, actions, returns) to posterior over task embedding.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        latent_dim: int = 8,
        sequence_length: int = 30
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        
        # Input processing
        # Trajectory: [state, action, return] at each timestep
        input_dim = state_dim + action_dim + 1  # +1 for portfolio return
        
        self.input_projection = nn.Linear(input_dim, hidden_size)
        
        # GRU for sequential processing
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # Output to latent distribution parameters
        self.mu_head = nn.Linear(hidden_size, latent_dim)
        self.logvar_head = nn.Linear(hidden_size, latent_dim)
        
    def forward(self, trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            trajectory: [batch_size, sequence_length, input_dim]
        Returns:
            mu: [batch_size, latent_dim]
            logvar: [batch_size, latent_dim]
        """
        batch_size, seq_len, input_dim = trajectory.shape
        
        # Project inputs
        x = torch.relu(self.input_projection(trajectory))  # [B, T, hidden]
        
        # Process with GRU
        gru_out, hidden = self.gru(x)  # gru_out: [B, T, hidden], hidden: [1, B, hidden]
        
        # Use final hidden state for task inference
        final_hidden = hidden.squeeze(0)  # [B, hidden]
        
        # Output distribution parameters
        mu = self.mu_head(final_hidden)      # [B, latent_dim]
        logvar = self.logvar_head(final_hidden)  # [B, latent_dim]
        
        return mu, logvar


class PortfolioDecoder(nn.Module):
    """
    Decoder that reconstructs portfolio returns from task embedding.
    """
    
    def __init__(
        self,
        latent_dim: int,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        prediction_horizon: int = 5
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.prediction_horizon = prediction_horizon
        
        # Combine task embedding with current state/action
        input_dim = latent_dim + state_dim + action_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Predict portfolio return
        )
        
    def forward(
        self, 
        task_embedding: torch.Tensor, 
        states: torch.Tensor, 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            task_embedding: [batch_size, latent_dim]
            states: [batch_size, sequence_length, state_dim]
            actions: [batch_size, sequence_length, action_dim]
        Returns:
            predicted_returns: [batch_size, sequence_length, 1]
        """
        batch_size, seq_len, _ = states.shape
        
        # Expand task embedding to match sequence length
        task_emb_expanded = task_embedding.unsqueeze(1).repeat(1, seq_len, 1)  # [B, T, latent_dim]
        
        # Concatenate inputs
        decoder_input = torch.cat([task_emb_expanded, states, actions], dim=-1)  # [B, T, total_dim]
        
        # Reshape for processing
        decoder_input_flat = decoder_input.view(-1, decoder_input.shape[-1])  # [B*T, total_dim]
        
        # Decode
        predicted_returns_flat = self.decoder(decoder_input_flat)  # [B*T, 1]
        
        # Reshape back
        predicted_returns = predicted_returns_flat.view(batch_size, seq_len, 1)  # [B, T, 1]
        
        return predicted_returns


class VariBADPortfolioPolicy(nn.Module):
    """
    Policy that conditions on task posterior for portfolio decisions.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_size: int = 256
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Policy network conditioned on state + task posterior
        input_dim = state_dim + 2 * latent_dim  # state + mu + logvar of posterior
        
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()  # Portfolio weights in [-1, 1]
        )
        
    def forward(self, state: torch.Tensor, task_mu: torch.Tensor, task_logvar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]
            task_mu: [batch_size, latent_dim]
            task_logvar: [batch_size, latent_dim]
        Returns:
            action: [batch_size, action_dim]
        """
        # Concatenate state with task posterior parameters
        policy_input = torch.cat([state, task_mu, task_logvar], dim=-1)
        
        action = self.policy_net(policy_input)
        
        return action


class VariBADPortfolioEnv:
    """
    VariBAD environment wrapper for portfolio optimization.
    Implements the meta-learning setup with task inference and belief updates.
    """
    
    def __init__(
        self,
        base_env: PortfolioEnv,
        encoder: PortfolioVariationalEncoder,
        decoder: PortfolioDecoder,
        policy: VariBADPortfolioPolicy,
        episode_length: int = 30,
        device: str = 'cpu'
    ):
        self.base_env = base_env
        self.encoder = encoder
        self.decoder = decoder
        self.policy = policy
        self.episode_length = episode_length
        self.device = device
        
        # Task inference state
        self.trajectory_buffer = []
        self.current_task_posterior = None
        self.current_step = 0
        
        # Prior for task embedding
        self.task_prior_mu = torch.zeros(encoder.latent_dim).to(device)
        self.task_prior_logvar = torch.zeros(encoder.latent_dim).to(device)
        
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset for new task/episode."""
        # Reset base environment
        obs = self.base_env.reset(seed)
        
        # Reset task inference
        self.trajectory_buffer = []
        self.current_step = 0
        
        # Initialize with prior
        self.current_task_posterior = {
            'mu': self.task_prior_mu.clone(),
            'logvar': self.task_prior_logvar.clone()
        }
        
        return self._augment_observation(obs)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute step with task belief update."""
        # Execute action in base environment
        obs, reward, done, info = self.base_env.step(action)
        
        # Store trajectory for task inference
        self._update_trajectory_buffer(obs, action, info['daily_return'])
        
        # Update task posterior
        self._update_task_posterior()
        
        self.current_step += 1
        
        # Check if episode is done (30 days)
        if self.current_step >= self.episode_length:
            done = True
        
        return self._augment_observation(obs), reward, done, info
    
    def _augment_observation(self, obs: np.ndarray) -> np.ndarray:
        """Add task posterior to observation."""
        # Convert task posterior to numpy and concatenate with observation
        task_mu = self.current_task_posterior['mu'].cpu().numpy()
        task_logvar = self.current_task_posterior['logvar'].cpu().numpy()
        
        augmented_obs = np.concatenate([obs, task_mu, task_logvar])
        return augmented_obs
    
    def _update_trajectory_buffer(self, obs: np.ndarray, action: np.ndarray, portfolio_return: float):
        """Update trajectory buffer for task inference."""
        # Store [obs, action, return] tuple
        trajectory_step = np.concatenate([obs, action, [portfolio_return]])
        self.trajectory_buffer.append(trajectory_step)
    
    def _update_task_posterior(self):
        """Update task posterior using encoder."""
        if len(self.trajectory_buffer) == 0:
            return
        
        # Convert trajectory buffer to tensor
        trajectory = np.stack(self.trajectory_buffer)  # [T, obs+action+return]
        trajectory_tensor = torch.FloatTensor(trajectory).unsqueeze(0).to(self.device)  # [1, T, dim]
        
        # Encode trajectory to get posterior
        with torch.no_grad():
            mu, logvar = self.encoder(trajectory_tensor)
            
            self.current_task_posterior = {
                'mu': mu.squeeze(0),  # [latent_dim]
                'logvar': logvar.squeeze(0)  # [latent_dim]
            }
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action from policy conditioned on task posterior."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        task_mu = self.current_task_posterior['mu'].unsqueeze(0)
        task_logvar = self.current_task_posterior['logvar'].unsqueeze(0)
        
        with torch.no_grad():
            action = self.policy(obs_tensor, task_mu, task_logvar)
            
        return action.squeeze(0).cpu().numpy()


class VariBADTrainer:
    """
    Trainer for VariBAD portfolio optimization.
    Implements the two-phase training: VAE learning + RL learning.
    """
    
    def __init__(
        self,
        encoder: PortfolioVariationalEncoder,
        decoder: PortfolioDecoder,
        policy: VariBADPortfolioPolicy,
        vae_lr: float = 1e-3,
        policy_lr: float = 1e-3,
        device: str = 'cpu'
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.policy = policy
        self.device = device
        
        # Separate optimizers as per paper
        self.vae_optimizer = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), 
            lr=vae_lr
        )
        self.policy_optimizer = optim.Adam(policy.parameters(), lr=policy_lr)
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        
    def compute_vae_loss(
        self,
        trajectories: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        beta: float = 1.0
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute VAE loss (reconstruction + KL divergence).
        
        Args:
            trajectories: [batch_size, seq_len, obs+action+return_dim]
            states: [batch_size, seq_len, state_dim]
            actions: [batch_size, seq_len, action_dim]
            returns: [batch_size, seq_len, 1]
            beta: KL regularization weight
        """
        batch_size, seq_len, _ = trajectories.shape
        
        # Encode trajectory to get posterior
        mu, logvar = self.encoder(trajectories)
        
        # Sample from posterior
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Decode to predict returns
        predicted_returns = self.decoder(z, states, actions)
        
        # Reconstruction loss
        recon_loss = self.mse_loss(predicted_returns, returns)
        
        # KL divergence loss (compare to unit Gaussian prior)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # Total VAE loss
        vae_loss = recon_loss + beta * kl_loss
        
        loss_dict = {
            'vae_loss': vae_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }
        
        return vae_loss, loss_dict
    
    def update_vae(self, batch_data: Dict) -> Dict:
        """Update VAE (encoder + decoder)."""
        self.vae_optimizer.zero_grad()
        
        # Compute VAE loss
        vae_loss, loss_dict = self.compute_vae_loss(
            batch_data['trajectories'],
            batch_data['states'],
            batch_data['actions'],
            batch_data['returns']
        )
        
        # Backward pass
        vae_loss.backward()
        self.vae_optimizer.step()
        
        return loss_dict


def create_varibad_from_loader(
    loader,
    split: str = 'train',
    latent_dim: int = 8,
    hidden_size: int = 128,
    episode_length: int = 30,
    device: str = 'cpu',
    **env_kwargs
) -> VariBADPortfolioEnv:
    """
    Create VariBAD environment from QuickSplitLoader.
    
    Args:
        loader: QuickSplitLoader instance
        split: 'train', 'val', or 'test'
        latent_dim: Dimensionality of task embedding
        hidden_size: Hidden layer size for networks
        episode_length: Episode length (30 days)
        device: torch device
    """
    # Create base environment
    from sp500_loader.core.environment import create_env_from_loader
    base_env = create_env_from_loader(
        loader,
        split=split,
        episode_length=episode_length,
        reward_mode='dsr',
        **env_kwargs
    )
    
    # Get dimensions
    state_dim = base_env.observation_space.shape[0]
    action_dim = base_env.action_space.shape[0]
    
    # Create VariBAD components
    encoder = PortfolioVariationalEncoder(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_size=hidden_size
    ).to(device)
    
    decoder = PortfolioDecoder(
        latent_dim=latent_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size
    ).to(device)
    
    policy = VariBADPortfolioPolicy(
        state_dim=state_dim + 2 * latent_dim,  # Augmented with task posterior
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_size=hidden_size * 2
    ).to(device)
    
    # Create VariBAD environment
    varibad_env = VariBADPortfolioEnv(
        base_env=base_env,
        encoder=encoder,
        decoder=decoder,
        policy=policy,
        episode_length=episode_length,
        device=device
    )
    
    return varibad_env


# Example usage and testing
if __name__ == "__main__":
    print("VariBAD Portfolio Optimization - Phase 1 Implementation")
    
    # This would be used with your existing data loader:
    """
    from sp500_loader import load_dataset, create_quick_loader
    
    # Load data
    panel_df = load_dataset('sp500_loader/data/sp500_dataset.parquet')
    loader = create_quick_loader(panel_df, episode_length=30)
    
    # Create VariBAD environment
    varibad_env = create_varibad_from_loader(
        loader,
        split='train',
        latent_dim=8,
        episode_length=30,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Test episode
    obs = varibad_env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    
    for step in range(5):
        action = varibad_env.get_action(obs)
        obs, reward, done, info = varibad_env.step(action)
        print(f"Step {step}: reward={reward:.6f}, done={done}")
        
        if done:
            break
    """
    
    print("Implementation ready for testing with your data!")
    print("\nNext steps:")
    print("1. Test with dummy data")
    print("2. Integrate with your SP500 data loader")
    print("3. Implement PPO training loop")
    print("4. Add task sampling mechanism")
    print("5. Compare with baseline methods")