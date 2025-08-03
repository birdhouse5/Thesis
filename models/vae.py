# In models/vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from logger_config import experiment_logger

logger = logging.getLogger(__name__)

class VAE(nn.Module):
    def __init__(self, obs_dim, action_dim=90, latent_dim=64, hidden_dim=256):
        super(VAE, self).__init__()
        
        self.obs_dim = obs_dim      # (30, num_features)
        self.action_dim = action_dim # For hierarchical: decisions + weights
        self.latent_dim = latent_dim
        
        # Encoder and decoders
        self.encoder = RNNEncoder(obs_dim, action_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.obs_decoder = ObservationDecoder(latent_dim, obs_dim, action_dim, hidden_dim)
        self.reward_decoder = RewardDecoder(latent_dim, obs_dim, action_dim, hidden_dim//2)


        logger.info(f"VAE initialized: latent_dim={latent_dim}, hidden_dim={hidden_dim}")

        # Log model architecture
        if experiment_logger:
            experiment_logger.log_hyperparams({
                'vae/latent_dim': latent_dim,
                'vae/hidden_dim': hidden_dim,
                'vae/obs_dim_0': obs_dim[0],
                'vae/obs_dim_1': obs_dim[1],
                'vae/action_dim': action_dim
            })

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for backpropagation through sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, obs_sequence, action_sequence, reward_sequence):
        """Encode trajectory to latent distribution"""
        mu, logvar, hidden_state = self.encoder(obs_sequence, action_sequence, reward_sequence)
        return mu, logvar, hidden_state
    
    def decode_obs(self, latent, current_obs, action):
        """Decode next observations"""
        return self.obs_decoder(latent, current_obs, action)
    
    def decode_reward(self, latent, current_obs, action, next_obs):
        """Decode reward"""
        return self.reward_decoder(latent, current_obs, action, next_obs)
    
    def forward(self, obs_seq, action_seq, reward_seq, return_latent=True):
        """Full VAE forward pass"""
        # Encode
        mu, logvar, hidden_state = self.encode(obs_seq, action_seq, reward_seq)
        
        # Sample latent
        latent = self.reparameterize(mu, logvar)
        
        # Log latent statistics
        if experiment_logger and self.training:
            experiment_logger.log_scalar('vae/latent_mean', mu.mean().item(), self.training_step)
            experiment_logger.log_scalar('vae/latent_std', torch.exp(0.5 * logvar).mean().item(), self.training_step)
            experiment_logger.log_histogram('vae/latent_distribution', latent, self.training_step)

        if return_latent:
            return latent, mu, logvar, hidden_state
        else:
            return mu, logvar, hidden_state
    
    # In models/vae.py - complete the compute_loss method
    def compute_loss(self, obs_seq, action_seq, reward_seq):
        """Compute VAE loss (reconstruction + KL divergence)"""
        batch_size, seq_len = obs_seq.shape[:2]
        
        # Encode sequence
        mu, logvar, _ = self.encode(obs_seq, action_seq, reward_seq)
        
        # Sample latent
        latent = self.reparameterize(mu, logvar)
        
        # Reconstruction losses
        recon_obs_loss = 0
        recon_reward_loss = 0
        
        # For each timestep, predict next observation and reward
        for t in range(seq_len - 1):
            current_obs = obs_seq[:, t]      # (batch, 30, features)
            current_action = action_seq[:, t] # (batch, 90)
            next_obs = obs_seq[:, t + 1]     # (batch, 30, features)
            reward = reward_seq[:, t]        # (batch, 1)
            
            # Predict next observation
            pred_next_obs = self.decode_obs(latent, current_obs, current_action)
            recon_obs_loss += F.mse_loss(pred_next_obs, next_obs)
            
            # Predict reward
            pred_reward = self.decode_reward(latent, current_obs, current_action, next_obs)
            recon_reward_loss += F.mse_loss(pred_reward, reward)
        
        # Average over sequence length
        recon_obs_loss /= (seq_len - 1)
        recon_reward_loss /= (seq_len - 1)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # Total loss
        total_loss = recon_obs_loss + recon_reward_loss + 0.1 * kl_loss  # Beta=0.1 for KL
        
        # Log individual components
        if experiment_logger and self.training:
            experiment_logger.log_scalars('vae/loss_components', {
                'reconstruction_obs': recon_obs_loss.item(),
                'reconstruction_reward': recon_reward_loss.item(),
                'kl_divergence': kl_loss.item(),
                'total': total_loss.item()
            }, self.training_step)
            
            self.training_step += 1
        
        return total_loss, {
            'recon_obs': recon_obs_loss.item(),
            'recon_reward': recon_reward_loss.item(), 
            'kl': kl_loss.item()
        }


class RNNEncoder(nn.Module):
    def __init__(self, obs_dim, action_dim, reward_dim=1, latent_dim=64, hidden_dim=256):
        super(RNNEncoder, self).__init__()
        
        self.obs_dim = obs_dim          # (30, num_features)
        self.action_dim = action_dim    # Flattened action representation
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Feature extractors for inputs
        obs_flat_dim = obs_dim[0] * obs_dim[1]  # 30 * num_features
        self.obs_encoder = nn.Linear(obs_flat_dim, 128)
        self.action_encoder = nn.Linear(action_dim, 64)
        self.reward_encoder = nn.Linear(reward_dim, 32)
        
        # RNN input dimension
        rnn_input_dim = 128 + 64 + 32  # obs + action + reward embeddings
        
        # GRU for sequential processing
        self.gru = nn.GRU(input_size=rnn_input_dim, 
                         hidden_size=hidden_dim, 
                         batch_first=True)
        
        # Output layers for latent distribution
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, obs_seq, action_seq, reward_seq, hidden_state=None):
        """
        Args:
            obs_seq: (batch, seq_len, 30, num_features)
            action_seq: (batch, seq_len, action_dim)
            reward_seq: (batch, seq_len, 1)
        """
        batch_size, seq_len = obs_seq.shape[:2]
        
        # Flatten observations
        obs_flat = obs_seq.view(batch_size, seq_len, -1)  # (batch, seq_len, 30*features)
        
        # Encode inputs
        obs_emb = F.relu(self.obs_encoder(obs_flat))      # (batch, seq_len, 128)
        action_emb = F.relu(self.action_encoder(action_seq))  # (batch, seq_len, 64)
        reward_emb = F.relu(self.reward_encoder(reward_seq))  # (batch, seq_len, 32)
        
        # Concatenate embeddings
        rnn_input = torch.cat([obs_emb, action_emb, reward_emb], dim=-1)  # (batch, seq_len, 224)
        
        # RNN forward pass
        rnn_output, hidden_state = self.gru(rnn_input, hidden_state)  # (batch, seq_len, hidden_dim)
        
        # Use final timestep for latent
        final_output = rnn_output[:, -1, :]  # (batch, hidden_dim)
        
        # Latent distribution parameters
        mu = self.fc_mu(final_output)        # (batch, latent_dim)
        logvar = self.fc_logvar(final_output)  # (batch, latent_dim)
        
        return mu, logvar, hidden_state
    

class ObservationDecoder(nn.Module):
    def __init__(self, latent_dim, obs_dim, action_dim=90, hidden_dim=256):
        super(ObservationDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim  # (30, num_features)
        self.action_dim = action_dim
        
        # Input: latent + current_obs + action
        obs_flat_dim = obs_dim[0] * obs_dim[1]
        input_dim = latent_dim + obs_flat_dim + action_dim
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_flat_dim)  # Predict next obs (flattened)
        )
        
    def forward(self, latent, current_obs, action):
        """
        Args:
            latent: (batch, latent_dim)
            current_obs: (batch, 30, num_features)
            action: (batch, 90)
        Returns:
            next_obs_pred: (batch, 30, num_features)
        """
        batch_size = latent.shape[0]
        
        # Flatten current obs
        obs_flat = current_obs.view(batch_size, -1)
        
        # Concatenate inputs
        decoder_input = torch.cat([latent, obs_flat, action], dim=-1)
        
        # Predict next observation (flattened)
        next_obs_flat = self.decoder(decoder_input)
        
        # Reshape back to original dimensions
        next_obs_pred = next_obs_flat.view(batch_size, self.obs_dim[0], self.obs_dim[1])
        
        return next_obs_pred

class RewardDecoder(nn.Module):
    def __init__(self, latent_dim, obs_dim, action_dim=90, hidden_dim=128):
        super(RewardDecoder, self).__init__()
        
        # Input: latent + current_obs + action + next_obs
        obs_flat_dim = obs_dim[0] * obs_dim[1]
        input_dim = latent_dim + 2 * obs_flat_dim + action_dim  # 2 obs for current + next
        
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Single reward value
        )
        
    def forward(self, latent, current_obs, action, next_obs):
        """
        Args:
            latent: (batch, latent_dim)
            current_obs: (batch, 30, num_features)
            action: (batch, 90)
            next_obs: (batch, 30, num_features)
        Returns:
            reward_pred: (batch, 1)
        """
        batch_size = latent.shape[0]
        
        # Flatten observations
        current_obs_flat = current_obs.view(batch_size, -1)
        next_obs_flat = next_obs.view(batch_size, -1)
        
        # Concatenate all inputs
        decoder_input = torch.cat([latent, current_obs_flat, action, next_obs_flat], dim=-1)
        
        # Predict reward
        reward_pred = self.decoder(decoder_input)
        
        return reward_pred