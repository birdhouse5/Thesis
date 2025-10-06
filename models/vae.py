import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)


class VAE(nn.Module):
    """
    Variational Autoencoder for learning task embeddings in VariBAD.
    Handles portfolio weight actions (30-dim) instead of hierarchical actions.
    """
    def __init__(self, obs_dim, num_assets, reward_dim=1, latent_dim=64, hidden_dim=256):
        super(VAE, self).__init__()
        
        self.obs_dim = obs_dim          # (N, F) - assets × features
        self.action_dim = num_assets    # Portfolio weights [N]
        self.reward_dim = reward_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Components
        self.encoder = RNNEncoder(self.obs_dim, self.action_dim, self.reward_dim, self.latent_dim, self.hidden_dim)
        self.obs_decoder = ObservationDecoder(self.latent_dim, self.obs_dim, self.action_dim, self.hidden_dim)
        self.reward_decoder = RewardDecoder(self.latent_dim, self.obs_dim, self.action_dim, self.hidden_dim//2)
        
        self.training_step = 0
    
    def reparameterize(self, mu, logvar):
        logvar_clamped = torch.clamp(logvar, min=-10, max=2)
        if (logvar != logvar_clamped).any():  # Only log when clipping occurred
            logger.debug(f"Clamped logvar - before: [{logvar.min():.2f}, {logvar.max():.2f}], "
                        f"after: [{logvar_clamped.min():.2f}, {logvar_clamped.max():.2f}]")
        logvar = logvar_clamped
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, obs_sequence, action_sequence, reward_sequence):
        """
        Encode trajectory to latent distribution.
        
        Args:
            obs_sequence: (batch, seq_len, N, F) - observation sequence
            action_sequence: (batch, seq_len, N) - portfolio weight sequence  
            reward_sequence: (batch, seq_len, 1) - reward sequence
            
        Returns:
            mu, logvar, hidden_state
        """
        return self.encoder(obs_sequence, action_sequence, reward_sequence)
    
    def decode_obs(self, latent, current_obs, action):
        """Decode next observation given current state and action."""
        return self.obs_decoder(latent, current_obs, action)
    
    def decode_reward(self, latent, current_obs, action, next_obs):
        """Decode reward given transition."""
        return self.reward_decoder(latent, current_obs, action, next_obs)
    
    def forward(self, obs_seq, action_seq, reward_seq, return_latent=True):
        """
        Full VAE forward pass.
        
        Args:
            obs_seq: (batch, seq_len, N, F) - observation sequence
            action_seq: (batch, seq_len, N) - portfolio weight sequence
            reward_seq: (batch, seq_len, 1) - reward sequence
            
        Returns:
            latent, mu, logvar, hidden_state
        """
        # Encode sequence to latent distribution
        mu, logvar, hidden_state = self.encode(obs_seq, action_seq, reward_seq)
        
        if return_latent:
            # Sample latent variable
            latent = self.reparameterize(mu, logvar)
            return latent, mu, logvar, hidden_state
        else:
            return mu, logvar, hidden_state
 
 
    def compute_loss(
        self,
        obs_seq, action_seq, reward_seq,
        beta=0.1,
        num_elbo_terms=8,
        prior_mu=None, prior_logvar=None  # Renamed from prev_* for clarity
    ):
        """
        VariBAD-style VAE loss with recursive prior.
        At each timestep t: encode τ:t, use previous posterior as prior.
        
        Args:
            obs_seq: (batch, H, N, F) - full observation trajectory
            action_seq: (batch, H, N) - full action trajectory
            reward_seq: (batch, H, 1) - full reward trajectory
            beta: KL divergence weight
            num_elbo_terms: number of timesteps to sample for ELBO computation
            prior_mu, prior_logvar: prior from previous episode (or None for N(0,I))
        
        Returns:
            total_loss: scalar loss
            info_dict: dictionary with loss components and diagnostics
        """
        B, H = obs_seq.shape[:2]
        device = obs_seq.device
        
        # Sample timesteps for ELBO computation
        if num_elbo_terms >= H:
            timesteps = list(range(1, H + 1))
        else:
            timesteps = torch.randint(1, H + 1, (num_elbo_terms,)).tolist()
            timesteps = sorted(list(set(timesteps)))
        
        # Accumulate losses
        total_recon_obs_loss = 0.0
        total_recon_reward_loss = 0.0
        total_kl_loss = 0.0
        
        # Track posterior for recursive prior
        current_prior_mu = prior_mu
        current_prior_logvar = prior_logvar
        last_mu, last_logvar = None, None
        
        # Compute ELBO at each sampled timestep with recursive prior
        for i, t in enumerate(timesteps):
            # Encode context up to timestep t (NO truncation)
            encode_obs = obs_seq[:, :t]
            encode_actions = action_seq[:, :t]
            encode_rewards = reward_seq[:, :t]
            
            mu, logvar, _ = self.encode(encode_obs, encode_actions, encode_rewards)
            latent = self.reparameterize(mu, logvar)
            
            # Decode FULL trajectory
            pred_next_obs = self.obs_decoder.forward_seq(
                latent, obs_seq[:, :-1], action_seq[:, :-1]
            )
            pred_rewards = self.reward_decoder.forward_seq(
                latent, obs_seq[:, :-1], action_seq[:, :-1], obs_seq[:, 1:]
            )
            
            # Reconstruction losses
            recon_obs_loss = F.mse_loss(pred_next_obs, obs_seq[:, 1:])
            recon_reward_loss = F.mse_loss(pred_rewards, reward_seq[:, 1:])
            
            # KL divergence with recursive prior
            if i == 0 and current_prior_mu is None:
                # First timestep of first episode: KL(q || N(0,I))
                kl_loss = -0.5 * torch.sum(
                    1 + logvar - mu.pow(2) - logvar.exp(), dim=1
                ).mean()
            else:
                # Recursive: KL(q(m|τ:t) || q(m|τ:t-1))
                # Use current_prior from previous timestep or previous episode
                if i == 0:
                    # First timestep of subsequent episode: use prior from last episode
                    prior_mu_use = current_prior_mu
                    prior_logvar_use = current_prior_logvar
                else:
                    # Within episode: use previous timestep's posterior
                    prior_mu_use = last_mu
                    prior_logvar_use = last_logvar
                
                var_post = logvar.exp()
                var_prior = prior_logvar_use.exp()
                kl_elements = (
                    var_post / var_prior
                    + (mu - prior_mu_use).pow(2) / var_prior
                    - 1
                    + prior_logvar_use - logvar
                )
                kl_loss = 0.5 * torch.sum(kl_elements, dim=1).mean()
            
            # Accumulate
            total_recon_obs_loss += recon_obs_loss
            total_recon_reward_loss += recon_reward_loss
            total_kl_loss += kl_loss
            
            # Update for next iteration
            last_mu, last_logvar = mu.detach(), logvar.detach()
        
        # Average over sampled timesteps
        num_terms = len(timesteps)
        avg_recon_obs = total_recon_obs_loss / num_terms
        avg_recon_reward = total_recon_reward_loss / num_terms
        avg_kl = total_kl_loss / num_terms
        
        total_loss = avg_recon_obs + avg_recon_reward + beta * avg_kl
        
        logger = logging.getLogger(__name__)
        # logger.info(f"=== VAE.compute_loss DEBUG ===")
        # logger.info(f"  Input shapes: obs={obs_seq.shape}, actions={action_seq.shape}, rewards={reward_seq.shape}")
        # logger.info(f"  Num ELBO terms requested: {num_elbo_terms}")
        # logger.info(f"  Timesteps sampled: {timesteps}")
        # logger.info(f"  Prior provided: {prior_mu is not None}")
        # logger.info(f"  Returning info dict keys: {list(info_dict.keys()) if 'info_dict' in locals() else 'NOT YET CREATED'}")

        return total_loss, {
            "total": total_loss.item(),
            "recon_obs": avg_recon_obs.item(),
            "recon_reward": avg_recon_reward.item(),
            "kl": avg_kl.item(),
            "num_elbo_terms": num_terms,
            "timesteps_sampled": timesteps,
            "latent_mu_mean": last_mu.mean().item() if last_mu is not None else 0.0,
            "latent_logvar_mean": last_logvar.mean().item() if last_logvar is not None else 0.0,
        }


class RNNEncoder(nn.Module):
    """
    RNN encoder for processing trajectory sequences.
    Encodes (observation, action, reward) sequences to latent distribution parameters.
    """
    def __init__(self, obs_dim, action_dim, reward_dim=1, latent_dim=64, hidden_dim=256):
        super(RNNEncoder, self).__init__()
        
        self.obs_dim = obs_dim          # (N, F)
        self.action_dim = action_dim    # N (portfolio weights)
        self.reward_dim = reward_dim    # 1 (scalar reward)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Input encoders
        obs_flat_dim = obs_dim[0] * obs_dim[1]  # N × F
        self.obs_encoder = nn.Linear(obs_flat_dim, 128)
        self.action_encoder = nn.Linear(action_dim, 64)      # Portfolio weights
        self.reward_encoder = nn.Linear(reward_dim, 32)
        
        # RNN for sequential processing
        rnn_input_dim = 128 + 64 + 32  # obs + action + reward embeddings
        self.gru = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Latent distribution outputs
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, obs_seq, action_seq, reward_seq, hidden_state=None):
        """
        Encode trajectory sequence to latent distribution.
        
        Args:
            obs_seq: (batch, seq_len, N, F) - observation sequence
            action_seq: (batch, seq_len, N) - portfolio weight sequence
            reward_seq: (batch, seq_len, 1) - reward sequence
            
        Returns:
            mu, logvar, hidden_state
        """

        batch_size, seq_len = obs_seq.shape[:2]
        
        # Ensure reward_seq is 2D [batch_size, seq_len]
        if reward_seq.dim() == 1:
            reward_seq = reward_seq.unsqueeze(0)  # [seq_len] → [1, seq_len]
        elif reward_seq.dim() == 3:
            reward_seq = reward_seq.squeeze(-1)   # [batch, seq_len, 1] → [batch, seq_len]
        
        # Now we're guaranteed 2D [batch_size, seq_len]
        assert reward_seq.shape == (batch_size, seq_len)
        
        # Encode inputs
        obs_flat = obs_seq.reshape(batch_size, seq_len, -1)
        obs_emb = F.relu(self.obs_encoder(obs_flat))
        action_emb = F.relu(self.action_encoder(action_seq))
        
        # Handle rewards properly for linear layer
        # Linear layer expects [*, input_features], so reshape to [batch*seq, 1]
        reward_flat = reward_seq.reshape(-1, 1)  # [batch*seq, 1]
        reward_emb_flat = F.relu(self.reward_encoder(reward_flat))  # [batch*seq, 32]
        reward_emb = reward_emb_flat.view(batch_size, seq_len, 32)  # [batch, seq, 32]
            
        # Concatenate embeddings
        rnn_input = torch.cat([obs_emb, action_emb, reward_emb], dim=-1)  # (batch, seq_len, 224)
        
        # Process through RNN
        rnn_output, hidden_state = self.gru(rnn_input, hidden_state)  # (batch, seq_len, hidden_dim)
        
        # Use final timestep for latent parameters
        final_output = rnn_output[:, -1, :]  # (batch, hidden_dim)
        
        # Generate latent distribution parameters
        mu = self.fc_mu(final_output)        # (batch, latent_dim)
        logvar = self.fc_logvar(final_output) # (batch, latent_dim)
        
        return mu, logvar, hidden_state


class ObservationDecoder(nn.Module):
    """
    Decoder for predicting next observations.
    """
    def __init__(self, latent_dim, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        obs_flat = obs_dim[0] * obs_dim[1]
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + obs_flat + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_flat),
        )
        self.obs_dim = obs_dim


    def forward_seq(self, latent, obs_seq, action_seq):
        """
        Decode sequence of next observations given obs and actions.
        Args:
            latent: (B, L) latent sample
            obs_seq: (B, T, N, F)
            action_seq: (B, T, N)
        Returns:
            pred_next_obs: (B, T, N, F)
        """
        B, T, N, F = obs_seq.shape
        latent_exp = latent.unsqueeze(1).expand(B, T, -1)  # (B, T, L)
        x = torch.cat([
            obs_seq.reshape(B, T, -1),
            action_seq.reshape(B, T, -1),
            latent_exp
        ], dim=-1)
        x = self.fc(x)   # implement with nn.Sequential
        return x.view(B, T, N, F)



class RewardDecoder(nn.Module):
    """
    Decoder for predicting rewards.
    """
    def __init__(self, latent_dim, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        obs_flat = obs_dim[0] * obs_dim[1]
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + obs_flat*2 + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    # def forward(self, latent, current_obs, action, next_obs):
    #     """
    #     Predict reward for transition.
        
    #     Args:
    #         latent: (batch, latent_dim) - task embedding
    #         current_obs: (batch, N, F) - current observation  
    #         action: (batch, N) - portfolio weights
    #         next_obs: (batch, N, F) - next observation
            
    #     Returns:
    #         reward_pred: (batch, 1) - predicted reward
    #     """
    #     batch_size = latent.shape[0]
        
    #     # Flatten observations
    #     current_obs_flat = current_obs.reshape(batch_size, -1)  # (batch, N×F)
    #     next_obs_flat = next_obs.reshape(batch_size, -1)        # (batch, N×F)
        
    #     # Concatenate all inputs
    #     decoder_input = torch.cat([latent, current_obs_flat, action, next_obs_flat], dim=-1)
        
    #     # Predict reward
    #     reward_pred = self.decoder(decoder_input)

    #     return reward_pred
        
    def forward_seq(self, latent, obs_seq, action_seq, next_obs_seq):
        """
        Decode rewards for a sequence.
        Returns: (B, T, 1)
        """
        B, T, N, F = obs_seq.shape
        latent_exp = latent.unsqueeze(1).expand(B, T, -1)
        x = torch.cat([
            obs_seq.reshape(B, T, -1),
            action_seq.reshape(B, T, -1),
            next_obs_seq.reshape(B, T, -1),
            latent_exp
        ], dim=-1)
        x = self.fc(x)
        return x  # (B, T, 1)