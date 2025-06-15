"""Trading policy that conditions on market belief."""

import torch
import torch.nn as nn

class TradingPolicy(nn.Module):
    """Generates portfolio weights conditioned on state and market belief."""
    
    def __init__(self, config):
        super().__init__()
        # TODO: Define state size based on features
        state_size = 50  # Placeholder
        latent_dim = config['model']['latent_dim']
        hidden_size = config['model']['policy']['hidden_size']
        n_assets = len(config['data']['assets'])
        
        self.mlp = nn.Sequential(
            nn.Linear(state_size + latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_assets)
        )
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, state, z):
        """
        Args:
            state: current market state
            z: sampled latent market regime
        Returns:
            portfolio weights (sum to 1)
        """
        x = torch.cat([state, z], dim=-1)
        logits = self.mlp(x)
        weights = self.softmax(logits)
        return weights
