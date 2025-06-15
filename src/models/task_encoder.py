"""Encoder that learns posterior over market regimes."""

import torch
import torch.nn as nn
from torch.distributions import Normal

class TaskEncoder(nn.Module):
    """Encodes trading trajectory into belief over market conditions."""
    
    def __init__(self, config):
        super().__init__()
        # TODO: Define input size based on state/action/reward dimensions
        input_size = 100  # Placeholder
        hidden_size = config['model']['encoder']['hidden_size']
        latent_dim = config['model']['latent_dim']
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=config['model']['encoder']['n_layers'],
            batch_first=True
        )
        
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_var = nn.Linear(hidden_size, latent_dim)
        
    def forward(self, trajectory):
        """
        Args:
            trajectory: (batch, seq_len, features)
        Returns:
            Normal distribution over latent market regime
        """
        _, hidden = self.gru(trajectory)
        hidden = hidden[-1]  # Last layer
        
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        std = torch.exp(0.5 * log_var)
        
        return Normal(mu, std)
