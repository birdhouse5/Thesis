import torch
import torch.nn as nn
import torch.nn.functional as F


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
        """Reparameterization trick for backpropagation through sampling."""
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
        obs_seq, action_seq, reward_seq,   # full trajectory τ
        context_len=None,                  # how much of τ to encode
        beta=0.1,
        prev_mu=None, prev_logvar=None     # recursive prior
    ):
        """
        VariBAD-style VAE loss:
        - Encode context τ[:t] to posterior q(m|τ[:t])
        - Decode full trajectory τ[:H]
        - KL divergence against recursive prior (q(m|τ[:t-1})) or N(0,I) at t=0
        """

        B, H = obs_seq.shape[:2]

        # Select context for encoder
        if context_len is not None:
            encode_obs     = obs_seq[:, :context_len]
            encode_actions = action_seq[:, :context_len]
            encode_rewards = reward_seq[:, :context_len]
        else:
            encode_obs, encode_actions, encode_rewards = obs_seq, action_seq, reward_seq

        # Encode to latent distribution
        mu, logvar, _ = self.encode(encode_obs, encode_actions, encode_rewards)

        # Sample latent
        latent = self.reparameterize(mu, logvar)

        # === Full trajectory reconstruction ===
        pred_next_obs = self.obs_decoder.forward_seq(
            latent, obs_seq[:, :-1], action_seq[:, :-1]
        )   # (B, H-1, N, F)

        pred_rewards = self.reward_decoder.forward_seq(
            latent, obs_seq[:, :-1], action_seq[:, :-1], obs_seq[:, 1:]
        )   # (B, H-1, 1)

        # Losses
        recon_obs_loss = F.mse_loss(pred_next_obs, obs_seq[:, 1:])
        recon_reward_loss = F.mse_loss(pred_rewards, reward_seq[:, 1:])

        # === Recursive KL divergence ===
        if prev_mu is not None and prev_logvar is not None:
            # KL(q_t || q_{t-1})
            var_post = logvar.exp()
            var_prior = prev_logvar.exp()
            kl_elements = (
                var_post / var_prior
                + (mu - prev_mu).pow(2) / var_prior
                - 1
                + prev_logvar - logvar
            )
            kl_loss = 0.5 * torch.sum(kl_elements, dim=1).mean()
        else:
            # KL(q || N(0,I))
            kl_loss = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp(), dim=1
            ).mean()

        # Total
        total_loss = recon_obs_loss + recon_reward_loss + beta * kl_loss

        # Return with diagnostics
        return total_loss, {
            "total": total_loss.item(),
            "recon_obs": recon_obs_loss.item(),
            "recon_reward": recon_reward_loss.item(),
            "kl": kl_loss.item(),
            "context_len": int(context_len) if context_len else H,
            "latent_mu_mean": mu.mean().item(),
            "latent_logvar_mean": logvar.mean().item(),
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

        
    # def forward(self, latent, current_obs, action):
    #     """
    #     Predict next observation.
        
    #     Args:
    #         latent: (batch, latent_dim) - task embedding
    #         current_obs: (batch, N, F) - current observation
    #         action: (batch, N) - portfolio weights
            
    #     Returns:
    #         next_obs_pred: (batch, N, F) - predicted next observation
    #     """
    #     batch_size = latent.shape[0]
        
    #     # Flatten current observation
    #     obs_flat = current_obs.reshape(batch_size, -1)  # (batch, N×F)
        
    #     # Concatenate inputs
    #     decoder_input = torch.cat([latent, obs_flat, action], dim=-1)
        
    #     # Predict next observation (flattened)
    #     next_obs_flat = self.decoder(decoder_input)
        
    #     # Reshape to original observation dimensions
    #     next_obs_pred = next_obs_flat.reshape(batch_size, self.obs_dim[0], self.obs_dim[1])
        
    #     return next_obs_pred

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