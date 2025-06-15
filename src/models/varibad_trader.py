"""Main VariBAD model combining encoder and policy."""

import torch
import torch.nn as nn
from torch.distributions import Normal

from .task_encoder import TaskEncoder
from .trading_policy import TradingPolicy

class VariBADTrader(nn.Module):
    """VariBAD model for adaptive portfolio optimization."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = TaskEncoder(config)
        self.policy = TradingPolicy(config)
        
        # Get latent dimension
        if 'model' in config:
            self.latent_dim = config['model']['latent_dim']
        else:
            self.latent_dim = 8
        
    def get_prior(self, batch_size=1):
        """Get prior distribution with correct batch dimension."""
        return Normal(
            torch.zeros(batch_size, self.latent_dim),
            torch.ones(batch_size, self.latent_dim)
        )
        
    def forward(self, state, trajectory=None):
        """
        Args:
            state: current market state (batch_size, state_dim)
            trajectory: past trajectory for inference (optional)
        Returns:
            portfolio_weights, posterior_distribution
        """
        batch_size = state.shape[0]
        
        if trajectory is None:
            # Beginning of episode - use prior
            posterior = self.get_prior(batch_size)
        else:
            # Update belief based on trajectory
            posterior = self.encoder(trajectory)
            
        # Sample latent market regime
        z = posterior.rsample()  # Shape: (batch_size, latent_dim)
        
        # Generate portfolio weights
        weights = self.policy(state, z)
        
        return weights, posterior
    
    def compute_elbo(self, trajectory, rewards):
        """Compute evidence lower bound for training encoder."""
        # TODO: Implement ELBO computation
        pass