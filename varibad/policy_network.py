"""
Policy Network for Portfolio VariBAD
Outputs portfolio weights conditioned on market state and task belief
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np

class PortfolioPolicy(nn.Module):
    """
    Policy network that outputs portfolio weights conditioned on:
    - Market state (technical indicators, prices)
    - Task belief (posterior over market regimes)
    """
    
    def __init__(
        self,
        state_dim: int,        # Market state dimension
        latent_dim: int,       # Task embedding dimension
        num_assets: int,       # Number of assets
        hidden_dim: int = 128,
        use_cash: bool = True, # Whether to include cash position
        allow_shorting: bool = False,
        transaction_cost: float = 0.001
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.num_assets = num_assets
        self.use_cash = use_cash
        self.allow_shorting = allow_shorting
        self.transaction_cost = transaction_cost
        
        # Total action dimension (assets + cash if used)
        self.action_dim = num_assets + (1 if use_cash else 0)
        
        # Input dimension: state + latent (mean and std of posterior)
        input_dim = state_dim + latent_dim * 2  # mu and sigma
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
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
        
        # Portfolio allocation strategy heads
        self.momentum_head = nn.Linear(hidden_dim // 2, num_assets)
        self.mean_reversion_head = nn.Linear(hidden_dim // 2, num_assets)
        
        # Strategy mixing weights
        self.strategy_mixer = nn.Linear(hidden_dim // 2, 3)  # momentum, mean_reversion, direct
        
    def forward(
        self,
        state: torch.Tensor,           # [batch, state_dim]
        task_mu: torch.Tensor,         # [batch, latent_dim] 
        task_logvar: torch.Tensor,     # [batch, latent_dim]
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Generate portfolio weights and value estimate.
        
        Args:
            state: Current market state
            task_mu: Mean of task posterior
            task_logvar: Log-variance of task posterior
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: Portfolio weights [batch, action_dim]
            value: State value estimate [batch, 1]
            info: Additional information dict
        """
        batch_size = state.shape[0]
        
        # Convert logvar to std
        task_std = torch.exp(0.5 * task_logvar)
        
        # Concatenate state with task belief (mu and std)
        x = torch.cat([state, task_mu, task_std], dim=-1)
        
        # Shared processing
        shared_features = self.shared_net(x)
        
        # Value prediction
        value = self.value_head(shared_features)
        
        # Portfolio strategies
        momentum_weights = torch.tanh(self.momentum_head(shared_features))
        mean_reversion_weights = torch.tanh(self.mean_reversion_head(shared_features))
        direct_weights = torch.tanh(self.portfolio_head(shared_features))
        
        # Strategy mixing
        strategy_logits = self.strategy_mixer(shared_features)
        strategy_weights = F.softmax(strategy_logits, dim=-1)
        
        # Combine strategies
        if self.use_cash:
            # Separate asset weights and cash
            momentum_assets = momentum_weights
            mean_reversion_assets = mean_reversion_weights  
            direct_assets = direct_weights[:, :-1]
            direct_cash = direct_weights[:, -1:]
            
            # Mix asset strategies
            mixed_assets = (
                strategy_weights[:, 0:1] * momentum_assets +
                strategy_weights[:, 1:2] * mean_reversion_assets +
                strategy_weights[:, 2:3] * direct_assets
            )
            
            # Combine with cash
            raw_weights = torch.cat([mixed_assets, direct_cash], dim=-1)
        else:
            # Only assets, no cash
            raw_weights = (
                strategy_weights[:, 0:1] * momentum_weights +
                strategy_weights[:, 1:2] * mean_reversion_weights +
                strategy_weights[:, 2:3] * direct_weights
            )
        
        # Normalize to sum to 1 (portfolio constraint)
        portfolio_weights = F.softmax(raw_weights, dim=-1)
        
        # Add noise for exploration if not deterministic
        if not deterministic and self.training:
            noise_std = 0.1 * torch.exp(-0.5 * task_logvar.mean(dim=-1, keepdim=True))
            noise = torch.randn_like(portfolio_weights) * noise_std
            portfolio_weights = portfolio_weights + noise
            portfolio_weights = F.softmax(portfolio_weights, dim=-1)  # Re-normalize
        
        info = {
            'strategy_weights': strategy_weights,
            'momentum_weights': momentum_assets if self.use_cash else momentum_weights,
            'mean_reversion_weights': mean_reversion_assets if self.use_cash else mean_reversion_weights,
            'task_uncertainty': task_std.mean(dim=-1),
            'raw_weights': raw_weights
        }
        
        return portfolio_weights, value, info
    
    def get_action(
        self,
        state: torch.Tensor,
        task_mu: torch.Tensor,
        task_logvar: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """Convenience method to get action only."""
        action, _, _ = self.forward(state, task_mu, task_logvar, deterministic)
        return action
    
    def compute_transaction_costs(
        self,
        new_weights: torch.Tensor,      # [batch, num_assets]
        old_weights: torch.Tensor       # [batch, num_assets]  
    ) -> torch.Tensor:
        """Compute transaction costs for portfolio rebalancing."""
        weight_changes = torch.abs(new_weights - old_weights)
        if self.use_cash:
            # Don't penalize cash changes as much
            asset_changes = weight_changes[:, :-1]
            cash_changes = weight_changes[:, -1:] * 0.1  # Lower cost for cash
            total_changes = torch.cat([asset_changes, cash_changes], dim=-1)
        else:
            total_changes = weight_changes
            
        transaction_costs = total_changes.sum(dim=-1) * self.transaction_cost
        return transaction_costs


class SimplePortfolioPolicy(nn.Module):
    """
    Simplified policy for faster experimentation.
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
        input_dim = state_dim + latent_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_assets)
        )
        
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        state: torch.Tensor,
        task_embedding: torch.Tensor  # Single sample from posterior
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple forward pass with sampled task embedding."""
        x = torch.cat([state, task_embedding], dim=-1)
        
        features = self.net[:-1](x)  # All but last layer
        
        # Portfolio weights (softmax for normalization) 
        logits = self.net[-1](features)
        weights = F.softmax(logits, dim=-1)
        
        # Value estimate
        value = self.value_head(features)
        
        return weights, value


# Test the policy
if __name__ == "__main__":
    # Example usage
    batch_size = 32
    state_dim = 50
    latent_dim = 8
    num_assets = 30
    
    policy = PortfolioPolicy(
        state_dim=state_dim,
        latent_dim=latent_dim,
        num_assets=num_assets,
        use_cash=True
    )
    
    # Mock inputs
    state = torch.randn(batch_size, state_dim)
    task_mu = torch.randn(batch_size, latent_dim)
    task_logvar = torch.randn(batch_size, latent_dim) * 0.5  # Smaller variance
    
    # Get action and value
    action, value, info = policy(state, task_mu, task_logvar)
    
    print(f"State shape: {state.shape}")
    print(f"Task mu shape: {task_mu.shape}")
    print(f"Task logvar shape: {task_logvar.shape}")
    print(f"Portfolio weights shape: {action.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Weights sum: {action.sum(dim=-1)}")  # Should be ~1.0
    
    print(f"\nStrategy weights shape: {info['strategy_weights'].shape}")
    print(f"Task uncertainty: {info['task_uncertainty'].mean():.4f}")
    
    # Test transaction costs
    old_weights = torch.rand(batch_size, num_assets + 1)
    old_weights = F.softmax(old_weights, dim=-1)
    
    transaction_costs = policy.compute_transaction_costs(action, old_weights)
    print(f"Transaction costs shape: {transaction_costs.shape}")
    print(f"Average transaction cost: {transaction_costs.mean():.6f}")
    
    # Test simple policy
    simple_policy = SimplePortfolioPolicy(state_dim, latent_dim, num_assets)
    task_sample = torch.randn(batch_size, latent_dim)
    simple_weights, simple_value = simple_policy(state, task_sample)
    
    print(f"\nSimple policy weights shape: {simple_weights.shape}")
    print(f"Simple policy weights sum: {simple_weights.sum(dim=-1)}")