"""Set up the model directory structure for VariBAD trading."""

import os

def create_file(filepath, content):
    """Create a file with given content."""
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Created: {filepath}")

# Create __init__.py
init_content = '''"""VariBAD model components for portfolio optimization."""

from .task_encoder import TaskEncoder
from .trading_policy import TradingPolicy
from .varibad_trader import VariBADTrader

__all__ = ['TaskEncoder', 'TradingPolicy', 'VariBADTrader']
'''

create_file('src/models/__init__.py', init_content)

# Create task_encoder.py
encoder_content = '''"""Encoder that learns posterior over market regimes."""

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
'''

create_file('src/models/task_encoder.py', encoder_content)

# Create trading_policy.py
policy_content = '''"""Trading policy that conditions on market belief."""

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
'''

create_file('src/models/trading_policy.py', policy_content)

# Create varibad_trader.py
trader_content = '''"""Main VariBAD model combining encoder and policy."""

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
        
        # Prior - standard normal
        self.prior = Normal(
            torch.zeros(config['model']['latent_dim']),
            torch.ones(config['model']['latent_dim'])
        )
        
    def forward(self, state, trajectory=None):
        """
        Args:
            state: current market state
            trajectory: past trajectory for inference (optional)
        Returns:
            portfolio_weights, posterior_distribution
        """
        if trajectory is None:
            # Beginning of episode - use prior
            posterior = self.prior
        else:
            # Update belief based on trajectory
            posterior = self.encoder(trajectory)
            
        # Sample latent market regime
        z = posterior.rsample()  # Reparameterization trick
        
        # Generate portfolio weights
        weights = self.policy(state, z)
        
        return weights, posterior
    
    def compute_elbo(self, trajectory, rewards):
        """Compute evidence lower bound for training encoder."""
        # TODO: Implement ELBO computation
        pass
'''

create_file('src/models/varibad_trader.py', trader_content)

print("\nModel structure created successfully!")
print("\nNext steps:")
print("1. Define exact state/action dimensions")
print("2. Implement ELBO computation")
print("3. Add market decoder if needed")
print("4. Create training loop")