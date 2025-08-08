import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import logging

logger = logging.getLogger(__name__)

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
        
        logger.info(f"Policy initialized: {num_assets} assets, obs_shape={obs_shape}, latent_dim={latent_dim}")
        
    def forward(self, obs, latent):
        """
        Forward pass through policy network.
        
        Args:
            obs: (batch, N, F) - market observations
            latent: (batch, latent_dim) - task embedding from VAE
            
        Returns:
            dict with 'portfolio_weights' and 'value'
        """
        batch_size = obs.shape[0]
        
        # Encode observations: flatten and process
        obs_flat = obs.view(batch_size, -1)  # (batch, N×F)
        obs_features = self.obs_encoder(obs_flat)  # (batch, hidden_dim//2)
        
        # Encode latent task embedding
        latent_features = self.latent_encoder(latent)  # (batch, hidden_dim//4)
        
        # Combine features
        combined = torch.cat([obs_features, latent_features], dim=-1)  # (batch, combined_dim)
        shared_features = self.shared_layers(combined)  # (batch, hidden_dim//2)
        
        # Generate outputs
        portfolio_logits = self.actor_head(shared_features)  # (batch, num_assets)
        value = self.critic_head(shared_features)            # (batch, 1)
        
        # Softmax normalization for valid portfolio weights
        portfolio_weights = F.softmax(portfolio_logits, dim=-1)  # (batch, num_assets)
        
        return {
            'portfolio_weights': portfolio_weights,  # Sum = 1, each weight ∈ [0,1]
            'portfolio_logits': portfolio_logits,    # Raw logits for PPO
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
                # Use softmax probabilities directly
                portfolio_weights = output['portfolio_weights']
            else:
                # Add small amount of exploration noise
                logits = output['portfolio_logits'] 
                # Temperature sampling for exploration
                temp = 1.0
                portfolio_weights = F.softmax(logits / temp, dim=-1)
            
            return portfolio_weights, output['value']
    
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
        



# Alternative: More principled Dirichlet distribution approach
class DirichletPortfolioPolicy(PortfolioPolicy):
    """
    Portfolio policy using Dirichlet distribution for portfolio weights.
    More principled than categorical approximation but computationally heavier.
    """
    
    def __init__(self, obs_shape, latent_dim, num_assets, hidden_dim=256):
        super().__init__(obs_shape, latent_dim, num_assets, hidden_dim)
        
        # Additional layer to output concentration parameters
        self.concentration_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, num_assets),
            nn.Softplus()  # Ensure positive concentrations
        )
    
    def forward(self, obs, latent):
        """Forward pass using Dirichlet distribution."""
        batch_size = obs.shape[0]
        
        # Encode inputs (same as base class)
        obs_flat = obs.view(batch_size, -1)
        obs_features = self.obs_encoder(obs_flat)
        latent_features = self.latent_encoder(latent)
        combined = torch.cat([obs_features, latent_features], dim=-1)
        shared_features = self.shared_layers(combined)
        
        # Generate concentration parameters for Dirichlet
        concentrations = self.concentration_head(shared_features) + 1.0  # Ensure > 1
        
        # Value function
        value = self.critic_head(shared_features)
        
        return {
            'concentrations': concentrations,
            'value': value
        }
    
    def act(self, obs, latent, deterministic=False):
        """Sample from Dirichlet distribution."""
        output = self.forward(obs, latent)
        concentrations = output['concentrations']
        
        if deterministic:
            # Use expected value of Dirichlet (normalized concentrations)
            portfolio_weights = concentrations / concentrations.sum(dim=-1, keepdim=True)
        else:
            # Sample from Dirichlet distribution
            dist = torch.distributions.Dirichlet(concentrations)
            portfolio_weights = dist.sample()
        
        return portfolio_weights, output['value']
    
    def evaluate_actions(self, obs, latent, actions):
        """Evaluate using proper Dirichlet log probability."""
        output = self.forward(obs, latent)
        concentrations = output['concentrations']
        
        # Create Dirichlet distribution
        dist = torch.distributions.Dirichlet(concentrations)
        
        # Compute log probability and entropy
        log_probs = dist.log_prob(actions).unsqueeze(-1)  # (batch, 1)
        entropy = dist.entropy().unsqueeze(-1)           # (batch, 1)
        
        values = output['value']
        
        return values, log_probs, entropy