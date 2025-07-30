"""
Policy Network for Portfolio VariBAD
Outputs portfolio weights conditioned on market state, task belief, and previous actions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class PortfolioPolicy(nn.Module):
    """
    Policy network that outputs portfolio weights conditioned on:
    - Market state (current observations)
    - Task embedding (market regime from encoder)
    - Previous portfolio weights (t-1)
    """
    
    def __init__(
        self,
        state_dim: int,        # Market state dimension
        latent_dim: int,       # Task embedding dimension
        num_assets: int,       # Number of assets
        hidden_dim: int = 128,
        use_cash: bool = True  # Whether to include cash position
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.num_assets = num_assets
        self.use_cash = use_cash
        
        # Action dimension (assets + cash if used)
        self.action_dim = num_assets + (1 if use_cash else 0)
        
        # Input: state + task_embedding + previous_action
        input_dim = state_dim + latent_dim + self.action_dim
        
        # Feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Portfolio weights head
        self.portfolio_head = nn.Linear(hidden_dim // 2, self.action_dim)
        
        # Value function head (for actor-critic)
        self.value_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(
        self,
        state: torch.Tensor,           # [batch, state_dim]
        task_embedding: torch.Tensor,  # [batch, latent_dim] - sampled from posterior
        prev_action: torch.Tensor,     # [batch, action_dim] - previous portfolio weights
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate portfolio weights and value estimate.
        
        Args:
            state: Current market state
            task_embedding: Sampled task embedding from encoder
            prev_action: Previous portfolio weights
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: Portfolio weights [batch, action_dim]
            value: State value estimate [batch, 1]
        """
        # Concatenate all inputs
        x = torch.cat([state, task_embedding, prev_action], dim=-1)
        
        # Extract features
        features = self.feature_net(x)
        
        # Generate portfolio logits
        logits = self.portfolio_head(features)
        
        # Softmax to ensure weights sum to 1
        portfolio_weights = F.softmax(logits, dim=-1)
        
        # Value estimate
        value = self.value_head(features)
        
        # Add exploration noise if not deterministic
        if not deterministic and self.training:
            noise = torch.randn_like(portfolio_weights) * 0.01
            portfolio_weights = portfolio_weights + noise
            portfolio_weights = F.softmax(portfolio_weights, dim=-1)  # Re-normalize
        
        return portfolio_weights, value
    
    def get_action(
        self,
        state: torch.Tensor,
        task_embedding: torch.Tensor,
        prev_action: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """Convenience method to get action only."""
        action, _ = self.forward(state, task_embedding, prev_action, deterministic)
        return action


class SimplePortfolioPolicy(nn.Module):
    """
    Even simpler policy without cash option.
    """
    
    def __init__(
        self,
        state_dim: int,
        latent_dim: int, 
        num_assets: int,
        hidden_dim: int = 64
    ):
        super().__init__()
        
        self.num_assets = num_assets
        input_dim = state_dim + latent_dim + num_assets  # prev_action = num_assets
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_assets)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        state: torch.Tensor,
        task_embedding: torch.Tensor,
        prev_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple forward pass."""
        x = torch.cat([state, task_embedding, prev_action], dim=-1)
        
        # Portfolio weights
        logits = self.net(x)
        weights = F.softmax(logits, dim=-1)
        
        # Value estimate
        value = self.value_net(x)
        
        return weights, value


# Test the policy
if __name__ == "__main__":
    # Example usage
    batch_size = 32
    state_dim = 50      # Market features
    latent_dim = 8      # Task embedding
    num_assets = 30     # Assets
    
    policy = PortfolioPolicy(
        state_dim=state_dim,
        latent_dim=latent_dim,
        num_assets=num_assets,
        use_cash=True
    )
    
    # Mock inputs
    state = torch.randn(batch_size, state_dim)
    task_embedding = torch.randn(batch_size, latent_dim)
    prev_action = torch.rand(batch_size, num_assets + 1)  # +1 for cash
    prev_action = F.softmax(prev_action, dim=-1)  # Normalize previous weights
    
    # Get action and value
    action, value = policy(state, task_embedding, prev_action)
    
    print(f"State shape: {state.shape}")
    print(f"Task embedding shape: {task_embedding.shape}")
    print(f"Previous action shape: {prev_action.shape}")
    print(f"Portfolio weights shape: {action.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Weights sum: {action.sum(dim=-1)}")  # Should be ~1.0
    
    # Test simple policy
    simple_policy = SimplePortfolioPolicy(state_dim, latent_dim, num_assets)
    prev_simple = torch.rand(batch_size, num_assets)
    prev_simple = F.softmax(prev_simple, dim=-1)
    
    simple_weights, simple_value = simple_policy(state, task_embedding, prev_simple)
    
    print(f"\nSimple policy weights shape: {simple_weights.shape}")
    print(f"Simple policy weights sum: {simple_weights.sum(dim=-1)}")