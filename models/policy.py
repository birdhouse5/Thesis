# File: models/policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PortfolioPolicy(nn.Module):
    def __init__(self, obs_dim, latent_dim, num_assets, hidden_dim=256):
        super(PortfolioPolicy, self).__init__()
        
        self.obs_dim = obs_dim      # (30, num_features)
        self.latent_dim = latent_dim
        self.num_assets = num_assets
        
        # Observation encoder
        obs_flat_dim = obs_dim[0] * obs_dim[1]  # Flatten market state
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Latent encoder
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Combined processing
        combined_dim = hidden_dim // 2 + hidden_dim // 4
        self.policy_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_assets)  # Raw logits
        )

                # Split final layer
        self.actor_head = nn.Linear(hidden_dim // 2, num_assets)      # Action logits
        self.critic_head = nn.Linear(hidden_dim // 2, 1)             # Value function
        
        # Learnable log std for action distribution
        self.log_std = nn.Parameter(torch.zeros(num_assets))
        
    def forward(self, state, latent):
        """
        Args:
            state: (batch, 30, num_features) - current market state
            latent: (batch, latent_dim) - task embedding
        Returns:
            portfolio_weights: (batch, 30) - portfolio allocation
        """
        batch_size = state.shape[0]
        
        # Encode observations
        state_flat = state.view(batch_size, -1)
        state_features = self.obs_encoder(state_flat)
        
        # Encode latent
        latent_features = self.latent_encoder(latent)
        
        # Combine and generate portfolio
        combined = torch.cat([state_features, latent_features], dim=-1)
        logits = self.policy_head(combined)
        
        # Softmax for valid portfolio weights
        portfolio_weights = F.softmax(logits, dim=-1)
        
        return portfolio_weights

    def act(self, obs, latent, deterministic=False):
        """Sample action from policy distributions"""
        output = self.forward(obs, latent)
        
        # Sample decisions: [long, short, neutral] per asset
        decision_probs = F.softmax(output['decision_logits'], dim=-1)
        if deterministic:
            decisions = torch.argmax(decision_probs, dim=-1)  # (batch, num_assets)
        else:
            decisions = torch.multinomial(decision_probs.view(-1, 3), 1).view(decision_probs.shape[:2])
        
        # Sample weights using softmax (ensure positive)
        long_probs = F.softmax(output['long_logits'], dim=-1)    # (batch, num_assets)
        short_probs = F.softmax(output['short_logits'], dim=-1)  # (batch, num_assets)
        
        if deterministic:
            long_weights = long_probs
            short_weights = short_probs
        else:
            # For continuous weights, we can add some noise or use the probabilities directly
            long_weights = long_probs
            short_weights = short_probs
        
        action = {
            'decisions': decisions,           # (batch, num_assets) - 0=long, 1=short, 2=neutral
            'long_weights': long_weights,    # (batch, num_assets) - softmax probabilities
            'short_weights': short_weights,  # (batch, num_assets) - softmax probabilities
            'decision_probs': decision_probs # For computing log probabilities later
        }
        
        value = output['value']
        
        # Log action statistics
        if experiment_logger is not None and not deterministic:
            decisions = action['decisions']
            long_count = (decisions == 0).sum().item()
            short_count = (decisions == 1).sum().item()
            neutral_count = (decisions == 2).sum().item()
            
            experiment_logger.log_scalars('policy/action_distribution', {
                'long_positions': long_count,
                'short_positions': short_count,
                'neutral_positions': neutral_count
            }, getattr(self, 'action_count', 0))
            
            experiment_logger.log_scalar('policy/value_estimate', value.mean().item(), 
                                    getattr(self, 'action_count', 0))

            self.action_count = getattr(self, 'action_count', 0) + 1

        return action, value

    
    def evaluate_actions(self, obs, latent, actions):
        """Evaluate actions for training - compute log probabilities and entropy"""
        output = self.forward(obs, latent)
        
        # Extract action components
        decisions = actions['decisions']           # (batch, num_assets)
        long_weights = actions['long_weights']     # (batch, num_assets) 
        short_weights = actions['short_weights']   # (batch, num_assets)
        
        # Compute log probabilities for decisions
        decision_probs = F.softmax(output['decision_logits'], dim=-1)  # (batch, num_assets, 3)
        decision_log_probs = F.log_softmax(output['decision_logits'], dim=-1)
        
        # Gather log probs for taken decisions
        decisions_one_hot = F.one_hot(decisions, num_classes=3).float()  # (batch, num_assets, 3)
        action_log_probs = (decision_log_probs * decisions_one_hot).sum(dim=-1)  # (batch, num_assets)
        action_log_probs = action_log_probs.sum(dim=-1, keepdim=True)  # (batch, 1) - sum across assets
        
        # Compute entropy for exploration
        decision_entropy = -(decision_probs * decision_log_probs).sum(dim=-1)  # (batch, num_assets)
        dist_entropy = decision_entropy.mean()  # Scalar
        
        # Value from forward pass
        value = output['value']  # (batch, 1)
        
        logger.debug(f"Action evaluation: log_prob {action_log_probs.mean().item():.3f}, "
                    f"entropy {dist_entropy.item():.3f}")
        
        return value, action_log_probs, dist_entropy