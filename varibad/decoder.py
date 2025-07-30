"""
VAE Decoder for Portfolio VariBAD
Predicts future market states and rewards given task embeddings and actions
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

class PortfolioDecoder(nn.Module):
    """
    Decoder that predicts market dynamics (states, rewards) given:
    - Current state
    - Action (portfolio weights)
    - Task embedding (market regime)
    """
    
    def __init__(
        self,
        state_dim: int,      # Market state dimension
        action_dim: int,     # Portfolio action dimension
        latent_dim: int,     # Task embedding dimension
        hidden_dim: int,
        num_assets: int  # Number of assets for reward prediction
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.num_assets = num_assets
        
        # Input dimension: state + action + task_embedding
        input_dim = state_dim + action_dim + latent_dim
        
        # Shared encoder
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # State prediction head (next market state)
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, state_dim)
        )
        
        # Reward prediction head (portfolio returns)
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Single reward value
        )
        
        # Asset return prediction head (for auxiliary loss)
        self.asset_returns_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_assets)
        )
        
    def forward(
        self,
        state: torch.Tensor,           # [batch, state_dim]
        action: torch.Tensor,          # [batch, action_dim] 
        task_embedding: torch.Tensor   # [batch, latent_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next state and reward given current state, action, and task.
        
        Args:
            state: Current market state
            action: Portfolio weights
            task_embedding: Inferred market regime embedding
            
        Returns:
            next_state: Predicted next market state
            reward: Predicted portfolio reward
            asset_returns: Predicted individual asset returns
        """
        # Concatenate inputs
        x = torch.cat([state, action, task_embedding], dim=-1)
        
        # Shared processing
        shared_features = self.shared_layers(x)
        
        # Predictions
        next_state = self.state_head(shared_features)
        reward = self.reward_head(shared_features)
        asset_returns = self.asset_returns_head(shared_features)
        
        return next_state, reward, asset_returns
    
    def predict_sequence(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,         # [batch, seq_len, action_dim]
        task_embedding: torch.Tensor,
        deterministic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict a sequence of states and rewards (for planning).
        
        Args:
            initial_state: Starting market state [batch, state_dim]
            actions: Sequence of actions [batch, seq_len, action_dim]
            task_embedding: Task embedding [batch, latent_dim]
            deterministic: Whether to use deterministic predictions
            
        Returns:
            states: Predicted states [batch, seq_len+1, state_dim]
            rewards: Predicted rewards [batch, seq_len]
        """
        batch_size, seq_len, _ = actions.shape
        
        states = [initial_state]
        rewards = []
        
        current_state = initial_state
        
        for t in range(seq_len):
            action_t = actions[:, t, :]
            
            # Predict next state and reward
            next_state, reward, _ = self.forward(
                current_state, action_t, task_embedding
            )
            
            states.append(next_state)
            rewards.append(reward)
            current_state = next_state
        
        states = torch.stack(states, dim=1)      # [batch, seq_len+1, state_dim]
        rewards = torch.stack(rewards, dim=1)    # [batch, seq_len, 1]
        
        return states, rewards


class RewardDecoder(nn.Module):
    """
    Specialized decoder for reward prediction only (lighter version).
    Used when we only need to predict portfolio returns.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int, 
        latent_dim: int,
        hidden_dim: int = 64
    ):
        super().__init__()
        
        input_dim = state_dim + action_dim + latent_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        task_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Predict reward only."""
        x = torch.cat([state, action, task_embedding], dim=-1)
        return self.network(x)


# Test the decoder
if __name__ == "__main__":
    # Example usage
    batch_size = 32
    state_dim = 50    # Market features
    action_dim = 30   # Portfolio weights (number of assets)
    latent_dim = 8    # Task embedding size
    hidden_dim = 128
    num_assets = 30
    
    decoder = PortfolioDecoder(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        num_assets=num_assets,
        hidden_dim=hidden_dim
    )
    
    # Mock inputs
    state = torch.randn(batch_size, state_dim)
    action = torch.randn(batch_size, action_dim)
    task_embedding = torch.randn(batch_size, latent_dim)
    
    # Single step prediction
    next_state, reward, asset_returns = decoder(state, action, task_embedding)
    
    print(f"State shape: {state.shape}")
    print(f"Action shape: {action.shape}")
    print(f"Task embedding shape: {task_embedding.shape}")
    print(f"Predicted next state shape: {next_state.shape}")
    print(f"Predicted reward shape: {reward.shape}")
    print(f"Predicted asset returns shape: {asset_returns.shape}")
    
    # Sequence prediction
    seq_len = 10
    actions_seq = torch.randn(batch_size, seq_len, action_dim)
    states_pred, rewards_pred = decoder.predict_sequence(
        state, actions_seq, task_embedding
    )
    
    print(f"\nSequence prediction:")
    print(f"Predicted states shape: {states_pred.shape}")
    print(f"Predicted rewards shape: {rewards_pred.shape}")
    
    # Test lightweight reward decoder
    reward_decoder = RewardDecoder(state_dim, action_dim, latent_dim)
    reward_only = reward_decoder(state, action, task_embedding)
    print(f"Reward-only prediction shape: {reward_only.shape}")