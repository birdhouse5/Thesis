import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import torch
import torch.nn as nn
import torch.nn.functional as F

class PortfolioPolicy(nn.Module):
    """
    Portfolio policy that outputs allocation weights over assets.
    Actions are portfolio weights ∈ [0,1]^N where sum ≤ 1 (remainder is cash).
    """
    def __init__(self, obs_shape, latent_dim, num_assets, hidden_dim=256, noise_factor, random_policy):
        super(PortfolioPolicy, self).__init__()
        
        self.obs_shape = obs_shape      # (N, F) - assets × features
        self.latent_dim = latent_dim    
        self.num_assets = num_assets
        self.hidden_dim = hidden_dim
        self.noise_factor = noise_factor
        self.random_policy = random_policy
        
        # Input dimensions
        obs_flat_dim = obs_shape[0] * obs_shape[1]  # N × F flattened
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Latent encoder
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Combined layers
        combined_dim = hidden_dim // 2 + hidden_dim // 4
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Output heads
        self.actor_head = nn.Linear(hidden_dim // 2, num_assets)    # Portfolio logits
        self.critic_head = nn.Linear(hidden_dim // 2, 1)            # Value function
        
    def forward(self, obs, latent):
        """
        Forward pass through policy network.
        
        Args:
            obs: (batch, N, F)
            latent: (batch, latent_dim)
        """
        assert obs.dim() == 3, f"Expected obs (B,N,F), got {tuple(obs.shape)}"
        assert latent.dim() == 2, f"Expected latent (B,L), got {tuple(latent.shape)}"

        batch_size = obs.shape[0]
        
        # Encode observations
        obs_flat = obs.reshape(batch_size, -1)
        obs_features = self.obs_encoder(obs_flat)
        
        # Encode latent
        latent_features = self.latent_encoder(latent)
        
        # Combine
        combined = torch.cat([obs_features, latent_features], dim=-1)
        shared_features = self.shared_layers(combined)
        
        # Outputs
        portfolio_logits = self.actor_head(shared_features)  # (B, N)
        value = self.critic_head(shared_features)            # (B, 1)
        
        return {
            'raw_actions': portfolio_logits,
            'value': value
        }
    
    def act(self, obs, latent, deterministic=False):
        """
        Sample action from policy.
        """

        batch_size = obs.shape[0]

        if self.random_policy:
            # Purely random portfolio weights (before normalization by env)
            # Shape: (B, num_assets)
            random_actions = torch.empty(batch_size, self.num_assets, device=obs.device).uniform_(-1.0, 1.0)
            # Return dummy values for critic
            values = torch.zeros(batch_size, 1, device=obs.device)
            return random_actions, values
            
        with torch.no_grad():
            output = self.forward(obs, latent)
            if deterministic:
                actions = output['raw_actions']
            else:
                noise = torch.randn_like(output['raw_actions']) * self.noise_factor
                actions = output['raw_actions'] + noise
            return actions, output['value']
    
    def evaluate_actions(self, obs, latent, actions):
        """
        Evaluate actions for PPO training.
        
        Returns:
            values: (batch, 1)
            log_probs: (batch, 1)
            entropy: (batch, 1)
        """
        output = self.forward(obs, latent)

        # Convert logits → probabilities
        portfolio_logits = output['raw_actions']
        portfolio_probs = torch.softmax(portfolio_logits, dim=-1)

        # Log probability of chosen action
        log_probs = torch.sum(actions * torch.log(portfolio_probs + 1e-8), dim=-1, keepdim=True)

        # Entropy of distribution
        entropy = -torch.sum(portfolio_probs * torch.log(portfolio_probs + 1e-8), dim=-1, keepdim=True)

        values = output['value']
        return values, log_probs, entropy

    
    def get_value(self, obs, latent):
        """Get state value without sampling action."""
        with torch.no_grad():
            output = self.forward(obs, latent)
            return output['value']