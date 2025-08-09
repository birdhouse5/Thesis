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
        
        logger.info(f"VAE initialized: action_dim={self.action_dim} (portfolio weights), "
                   f"latent_dim={latent_dim}, obs_dim={obs_dim}")
        
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
    
    def compute_loss(self, obs_seq, action_seq, reward_seq, beta=0.1, context_len=None):
        """
        Compute VAE loss (reconstruction + KL divergence).
        If context_len provided, encode only τ:context_len but decode full sequence (VariBAD approach).
        
        Args:
            obs_seq: (batch, seq_len, N, F) - observation sequence
            action_seq: (batch, seq_len, N) - portfolio weight sequence  
            reward_seq: (batch, seq_len, 1) - reward sequence
            beta: Weight for KL divergence term
            context_len: If provided, encode only first context_len timesteps
            
        Returns:
            total_loss, loss_components_dict
        """
        batch_size, seq_len = obs_seq.shape[:2]
        
        # Determine what to encode vs decode
        if context_len is not None:
            # VariBAD: encode τ:context_len, decode full sequence
            encode_obs = obs_seq[:, :context_len]
            encode_actions = action_seq[:, :context_len]
            encode_rewards = reward_seq[:, :context_len]
            decode_obs = obs_seq
            decode_actions = action_seq
            decode_rewards = reward_seq
        else:
            # Standard VAE: encode and decode same sequence
            encode_obs = obs_seq
            encode_actions = action_seq
            encode_rewards = reward_seq
            decode_obs = obs_seq
            decode_actions = action_seq
            decode_rewards = reward_seq
        
        # Encode to latent distribution
        mu, logvar, _ = self.encode(encode_obs, encode_actions, encode_rewards)
        
        # Sample latent
        latent = self.reparameterize(mu, logvar)
        
        # Reconstruction losses - always decode full sequence
        recon_obs_loss = 0
        recon_reward_loss = 0
        
        # Predict next observations and rewards for each timestep
        decode_seq_len = decode_obs.shape[1]
        for t in range(decode_seq_len - 1):
            current_obs = decode_obs[:, t]      # (batch, N, F)
            current_action = decode_actions[:, t] # (batch, N)  
            next_obs = decode_obs[:, t + 1]     # (batch, N, F)
            reward = decode_rewards[:, t + 1]    # (batch, 1) - reward at t+1
            
            # Predict next observation
            pred_next_obs = self.decode_obs(latent, current_obs, current_action)
            recon_obs_loss += F.mse_loss(pred_next_obs, next_obs)
            
            # Predict reward  
            pred_reward = self.decode_reward(latent, current_obs, current_action, next_obs)
            recon_reward_loss += F.mse_loss(pred_reward, reward)
        
        # Average over sequence length
        recon_obs_loss /= (decode_seq_len - 1)
        recon_reward_loss /= (decode_seq_len - 1)
        
        # KL divergence with standard normal prior
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # Total loss
        total_loss = recon_obs_loss + recon_reward_loss + beta * kl_loss
        
        # Logging
        if self.training:
            self.training_step += 1
            
        loss_components = {
            'recon_obs': recon_obs_loss.item(),
            'recon_reward': recon_reward_loss.item(),
            'kl': kl_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_components


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
        
        # defensive reshape
        reward_seq = reward_seq.unsqueeze(-1) if reward_seq.dim()==2 else reward_seq
        reward_flat = reward_seq.reshape(batch_size * seq_len, self.reward_dim)


        # Flatten observations for encoding
        obs_flat = obs_seq.reshape(batch_size, seq_len, -1)  # (batch, seq_len, N×F)
        
        # Encode inputs - ensure correct shapes
        obs_emb = F.relu(self.obs_encoder(obs_flat))        # (batch, seq_len, 128)
        action_emb = F.relu(self.action_encoder(action_seq)) # (batch, seq_len, 64)
        
        # Reward encoding - handle shape carefully
        # reward_seq is (batch, seq_len, 1), we need to keep this shape for linear layer
        reward_flat = reward_seq.view(batch_size * seq_len, self.reward_dim)  # (batch*seq_len, 1)
        reward_emb_flat = F.relu(self.reward_encoder(reward_flat))  # (batch*seq_len, 32)
        reward_emb = reward_emb_flat.view(batch_size, seq_len, 32)  # (batch, seq_len, 32)
        
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
        super(ObservationDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim  # (N, F)
        self.action_dim = action_dim  # N
        
        # Input: latent + current_obs + action
        obs_flat_dim = obs_dim[0] * obs_dim[1]  # N × F
        input_dim = latent_dim + obs_flat_dim + action_dim
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_flat_dim)  # Predict flattened next observation
        )
        
    def forward(self, latent, current_obs, action):
        """
        Predict next observation.
        
        Args:
            latent: (batch, latent_dim) - task embedding
            current_obs: (batch, N, F) - current observation
            action: (batch, N) - portfolio weights
            
        Returns:
            next_obs_pred: (batch, N, F) - predicted next observation
        """
        batch_size = latent.shape[0]
        
        # Flatten current observation
        obs_flat = current_obs.reshape(batch_size, -1)  # (batch, N×F)
        
        # Concatenate inputs
        decoder_input = torch.cat([latent, obs_flat, action], dim=-1)
        
        # Predict next observation (flattened)
        next_obs_flat = self.decoder(decoder_input)
        
        # Reshape to original observation dimensions
        next_obs_pred = next_obs_flat.reshape(batch_size, self.obs_dim[0], self.obs_dim[1])
        
        return next_obs_pred


class RewardDecoder(nn.Module):
    """
    Decoder for predicting rewards.
    """
    def __init__(self, latent_dim, obs_dim, action_dim, hidden_dim=128):
        super(RewardDecoder, self).__init__()
        
        # Input: latent + current_obs + action + next_obs
        obs_flat_dim = obs_dim[0] * obs_dim[1]  # N × F
        input_dim = latent_dim + 2 * obs_flat_dim + action_dim  # Current + next obs
        
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Single reward value
        )
        
    def forward(self, latent, current_obs, action, next_obs):
        """
        Predict reward for transition.
        
        Args:
            latent: (batch, latent_dim) - task embedding
            current_obs: (batch, N, F) - current observation  
            action: (batch, N) - portfolio weights
            next_obs: (batch, N, F) - next observation
            
        Returns:
            reward_pred: (batch, 1) - predicted reward
        """
        batch_size = latent.shape[0]
        
        # Flatten observations
        current_obs_flat = current_obs.reshape(batch_size, -1)  # (batch, N×F)
        next_obs_flat = next_obs.reshape(batch_size, -1)        # (batch, N×F)
        
        # Concatenate all inputs
        decoder_input = torch.cat([latent, current_obs_flat, action, next_obs_flat], dim=-1)
        
        # Predict reward
        reward_pred = self.decoder(decoder_input)

        return reward_pred
        