import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

 

class PortfolioPolicy(nn.Module):
    def __init__(self, obs_shape, latent_dim, num_assets,
                 hidden_dim=256, noise_factor=0.0, random_policy=False,
                 action_scale=1.0,
                 min_logstd=-3.0, max_logstd=-0.3,
                 long_only=True):
                 
        super().__init__()
        self.obs_shape = obs_shape
        self.latent_dim = latent_dim
        self.num_assets = num_assets
        self.hidden_dim = hidden_dim
        self.noise_factor = noise_factor
        self.random_policy = random_policy
        self.action_scale = action_scale
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd
        self.long_only = long_only

        obs_flat_dim = obs_shape[0] * obs_shape[1]

        # Encoders
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim if latent_dim > 0 else 1, hidden_dim // 4),
            nn.ReLU()
        )

        combined_dim = hidden_dim // 2 + hidden_dim // 4
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Actor: mean + log_std for Gaussian
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim // 2, num_assets),
        )
        # self.actor_logstd = nn.Parameter(torch.zeros(num_assets)) TODO
        #self.actor_logstd = nn.Parameter(torch.zeros(num_assets) * -1.0)
        #self.actor_logstd = nn.Parameter(torch.zeros(num_assets))
        self.actor_logstd_head = nn.Linear(hidden_dim // 2, num_assets)

        # Critic
        self.critic_head = nn.Linear(hidden_dim // 2, 1)           # Value function
        
    def forward(self, obs, latent):
        B = obs.shape[0]
        obs_flat = obs.reshape(B, -1)
        obs_features = self.obs_encoder(obs_flat)

        if latent.shape[1] == 0:  # handle latent_dim=0
            latent = torch.zeros(B, 1, device=obs.device)

        # latent normalization TODO 
        latent = (latent - latent.mean(dim=0, keepdim=True)) / (latent.std(dim=0, keepdim=True) + 1e-6)
        latent = torch.nan_to_num(latent, nan=0.0, posinf=0.0, neginf=0.0)
        latent = torch.clamp(latent, -10, 10)

        latent_features = self.latent_encoder(latent)
        combined = torch.cat([obs_features, latent_features], dim=-1)
        shared = self.shared_layers(combined)

        mean = self.actor_mean(shared) * self.action_scale
        raw_logstd = self.actor_logstd_head(shared)
        min_logstd, max_logstd = -3.0, -0.3
        logstd_clamped = self.min_logstd + (self.max_logstd - self.min_logstd) * torch.sigmoid(raw_logstd)


        value = self.critic_head(shared)
        return mean, logstd_clamped.expand_as(mean), value

    def act(self, obs, latent, deterministic=False):

        mean, logstd, value = self.forward(obs, latent)
        std = logstd.exp()
        dist = Normal(mean, std)

        # Sample actions
        raw_actions = mean if deterministic else dist.rsample()

        # Define epsilon for numerical stability
        eps = 1e-8  # ADD THIS LINE


        # Map to portfolio weights
        bounded = torch.sigmoid(raw_actions)  # (0, 1) for long-only
        sum_weights = torch.sum(bounded, dim=-1, keepdim=True)
        sum_weights = sum_weights + eps
        weights = bounded / sum_weights

        # if self.long_only:
        #     bounded = torch.sigmoid(raw_actions)  # (0, 1) for long-only
        #     sum_weights = torch.sum(bounded, dim=-1, keepdim=True)
        #     sum_weights = sum_weights + eps
        #     weights = bounded / sum_weights
        # else:
        #     bounded = torch.tanh(raw_actions)  # (-1, 1) for long-short
        #     abs_sum = torch.sum(torch.abs(bounded), dim=-1, keepdim=True)
        #     abs_sum = abs_sum + eps
        #     weights = bounded / abs_sum

        log_prob = dist.log_prob(raw_actions).sum(-1, keepdim=True)

        return weights, value, log_prob, raw_actions

    def evaluate_actions(self, obs, latent, raw_actions):
        mean, logstd, value = self.forward(obs, latent)
        std = logstd.exp()
        dist = Normal(mean, std)
        log_probs = dist.log_prob(raw_actions).sum(-1, keepdim=True)
        entropy = dist.entropy().mean(-1, keepdim=True)  # normalize per dimension

        return value, log_probs, entropy