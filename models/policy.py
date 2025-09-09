import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class PortfolioPolicy(nn.Module):
    """
    Portfolio policy that outputs allocation weights over assets.
    Actions are portfolio weights ∈ [0,1]^N where sum ≤ 1 (remainder is cash).
    """
    def __init__(self, obs_shape, latent_dim, num_assets, hidden_dim=256):
        super(PortfolioPolicy, self).__init__()
        
        self.obs_shape = obs_shape      # (N, F) - assets × features
        self.latent_dim = latent_dim    
        self.num_assets = num_assets
        self.hidden_dim = hidden_dim
        
        # Input dimensions
        obs_flat_dim = obs_shape[0] * obs_shape[1]  # N × F flattened
        
        # Observation encoder: flatten market state and encode
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Latent encoder: encode task embedding
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Combined feature processing
        combined_dim = hidden_dim // 2 + hidden_dim // 4
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Output heads
        self.actor_head = nn.Linear(hidden_dim // 2, num_assets)    # Portfolio logits
        self.critic_head = nn.Linear(hidden_dim // 2, 1)           # Value function
        
    def forward(self, obs, latent):
        """
        Forward pass through policy network.
        
        Args:
            obs: (batch, N, F) - market observations
            latent: (batch, latent_dim) - task embedding from VAE
            
        Returns:
            dict with 'portfolio_weights' and 'value'
        """

        assert obs.dim() == 3, f"Expected obs (B,N,F), got {tuple(obs.shape)}"
        assert latent.dim() == 2, f"Expected latent (B,L), got {tuple(latent.shape)}"

        batch_size = obs.shape[0]
        
        # Encode observations: flatten and process
        obs_flat = obs.reshape(batch_size, -1)  # (batch, N×F)
        obs_features = self.obs_encoder(obs_flat)  # (batch, hidden_dim//2)
        
        # Encode latent task embedding
        latent_features = self.latent_encoder(latent)  # (batch, hidden_dim//4)
        
        # Combine features
        combined = torch.cat([obs_features, latent_features], dim=-1)  # (batch, combined_dim)
        shared_features = self.shared_layers(combined)  # (batch, hidden_dim//2)
        
        # Generate outputs
        portfolio_logits = self.actor_head(shared_features)  # (batch, num_assets)
        value = self.critic_head(shared_features)            # (batch, 1)
        
        return {
            'raw_actions': portfolio_logits,  # raw outputs, unconstrained
            'value': value
        }
    
    def act(self, obs, latent, deterministic=False):
        """
        Sample action from policy.
        
        Args:
            obs: (batch, N, F) - market observations  
            latent: (batch, latent_dim) - task embedding
            deterministic: If True, return mean action
            
        Returns:
            portfolio_weights: (batch, num_assets) - allocation weights
            value: (batch, 1) - state value estimate
        """
        with torch.no_grad():
            output = self.forward(obs, latent)
            
            if deterministic:
                actions = output['raw_actions']
            else:

                # add small noise for stochastic exploration
                noise = torch.randn_like(output['raw_actions']) * 0.01
                actions = output['raw_actions'] + noise
            return actions, output['value']

    
    def evaluate_actions(self, obs, latent, actions):
        """
        Evaluate actions for PPO training.
        
        Args:
            obs: (batch, N, F) - market observations
            latent: (batch, latent_dim) - task embeddings  
            actions: (batch, num_assets) - portfolio weights to evaluate
            
        Returns:
            values: (batch, 1) - state values
            log_probs: (batch, 1) - log probabilities of actions
            entropy: (batch, 1) - policy entropy
        """
        output = self.forward(obs, latent)
        
        # Use Dirichlet distribution for portfolio weights
        # Convert softmax outputs to Dirichlet concentration parameters
        alpha = output['portfolio_weights'] * 10.0 + 1e-8  # Concentration parameters
        
        # For simplicity, approximate with categorical distribution over assets
        # and compute log probability of the taken allocation
        portfolio_probs = output['portfolio_weights']  # (batch, num_assets)
        
        # Compute log probability: treat as weighted categorical
        # This is an approximation - ideally we'd use Dirichlet distribution
        log_probs = torch.sum(actions * torch.log(portfolio_probs + 1e-8), dim=-1, keepdim=True)
        
        # Entropy of categorical distribution
        entropy = -torch.sum(portfolio_probs * torch.log(portfolio_probs + 1e-8), dim=-1, keepdim=True)
        
        values = output['value']
        
        return values, log_probs, entropy
    
    def get_value(self, obs, latent):
        """Get state value without sampling action."""
        with torch.no_grad():
            output = self.forward(obs, latent)
            return output['value']