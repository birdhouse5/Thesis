# In models/policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from logger_config import experiment_logger

logger = logging.getLogger(__name__)

class Policy(nn.Module):
    def __init__(self, obs_shape, latent_dim, num_assets, algorithm='ppo', hidden_dim=256):
        super(Policy, self).__init__()
        
        self.num_assets = num_assets
        self.obs_shape = obs_shape
        self.algorithm = algorithm
        
        # Input processing
        input_dim = obs_shape[0] * obs_shape[1] + latent_dim
        
        # Shared feature extraction
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Decision head: [long, short, neutral] per asset
        self.decision_head = nn.Linear(hidden_dim, num_assets * 3)
        
        # Weight heads
        self.long_weight_head = nn.Linear(hidden_dim, num_assets)
        self.short_weight_head = nn.Linear(hidden_dim, num_assets)
        
        # Value head (used by both PPO and A2C)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # PPO-specific: we might need action log probabilities
        # A2C-specific: direct policy gradients

        logger.info(f"Policy initialized: assets={num_assets}, hidden_dim={hidden_dim}")
        
        # Log model architecture
        if experiment_logger:
            experiment_logger.log_hyperparams({
                'policy/num_assets': num_assets,
                'policy/hidden_dim': hidden_dim,
                'policy/latent_dim': latent_dim,
                'policy/obs_shape_0': obs_shape[0],
                'policy/obs_shape_1': obs_shape[1]
            })
        
    def forward(self, obs, latent):
        """Forward pass through hierarchical policy"""
        # Flatten observations and concatenate with latent
        batch_size = obs.shape[0]
        obs_flat = obs.view(batch_size, -1)  # Flatten (batch, 30, features) -> (batch, 30*features)
        
        if latent is not None:
            x = torch.cat([obs_flat, latent], dim=-1)
        else:
            x = obs_flat
        
        # Shared feature extraction
        shared_features = self.shared_net(x)
        
        # Decision logits: reshape to (batch, num_assets, 3)
        decision_logits = self.decision_head(shared_features)
        decision_logits = decision_logits.view(batch_size, self.num_assets, 3)
        
        # Weight logits
        long_logits = self.long_weight_head(shared_features)    # (batch, num_assets)
        short_logits = self.short_weight_head(shared_features)  # (batch, num_assets)
        
        # Value estimate
        value = self.value_head(shared_features)  # (batch, 1)
        
        return {
            'decision_logits': decision_logits,
            'long_logits': long_logits, 
            'short_logits': short_logits,
            'value': value
        }

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
        if experiment_logger and not deterministic:
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