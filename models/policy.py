import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from env import normalize_with_budget_constraint  # import helper

class PortfolioPolicy(nn.Module):
    def __init__(self, obs_shape, latent_dim, num_assets,
                 hidden_dim=256, noise_factor=0.0, random_policy=False):
        super().__init__()
        self.obs_shape = obs_shape
        self.latent_dim = latent_dim
        self.num_assets = num_assets
        self.hidden_dim = hidden_dim
        self.noise_factor = noise_factor
        self.random_policy = random_policy

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
        self.actor_mean = nn.Linear(hidden_dim // 2, num_assets)
        self.actor_logstd = nn.Parameter(torch.zeros(num_assets))
        # Critic
        self.critic_head = nn.Linear(hidden_dim // 2, 1)           # Value function
        
        def forward(self, obs, latent):
        B = obs.shape[0]
        obs_flat = obs.reshape(B, -1)
        obs_features = self.obs_encoder(obs_flat)

        if latent.shape[1] == 0:  # handle latent_dim=0
            latent = torch.zeros(B, 1, device=obs.device)

        latent_features = self.latent_encoder(latent)
        combined = torch.cat([obs_features, latent_features], dim=-1)
        shared = self.shared_layers(combined)

        mean = self.actor_mean(shared)
        value = self.critic_head(shared)
        return mean, self.actor_logstd.expand_as(mean), value

    def act(self, obs, latent, deterministic=False):
        mean, logstd, value = self.forward(obs, latent)
        std = logstd.exp()

        dist = Normal(mean, std)
        if deterministic:
            raw_actions = mean
        else:
            raw_actions = dist.rsample()  # reparameterized sample

        # Normalize to valid portfolio weights
        actions_np = raw_actions.detach().cpu().numpy()
        weights, _ = zip(*[normalize_with_budget_constraint(a) for a in actions_np])
        actions = torch.tensor(weights, dtype=torch.float32, device=obs.device)

        log_prob = dist.log_prob(raw_actions).sum(-1, keepdim=True)
        return actions, value, log_prob

    def evaluate_actions(self, obs, latent, raw_actions):
        mean, logstd, value = self.forward(obs, latent)
        std = logstd.exp()
        dist = Normal(mean, std)
        log_probs = dist.log_prob(raw_actions).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1).mean()
        return value, log_probs, entropy