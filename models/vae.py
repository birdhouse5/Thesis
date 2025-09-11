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
    
    def compute_loss(self, obs_seq, action_seq, reward_seq, beta=0.1, context_len=None):
        """
        Compute VAE loss (reconstruction + KL divergence) with enhanced component tracking.
        If context_len provided, encode only τ:context_len but decode full sequence (VariBAD approach).
        
        Args:
            obs_seq: (batch, seq_len, N, F) - observation sequence
            action_seq: (batch, seq_len, N) - portfolio weight sequence  
            reward_seq: (batch, seq_len, 1) - reward sequence
            beta: Weight for KL divergence term
            context_len: If provided, encode only first context_len timesteps
            
        Returns:
            total_loss, enhanced_loss_components_dict
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
        
        # === NEW: Track per-timestep losses for analysis ===
        obs_losses_per_step = []
        reward_losses_per_step = []
        
        # Predict next observations and rewards for each timestep
        decode_seq_len = decode_obs.shape[1]
        for t in range(decode_seq_len - 1):
            current_obs = decode_obs[:, t]      # (batch, N, F)
            current_action = decode_actions[:, t] # (batch, N)  
            next_obs = decode_obs[:, t + 1]     # (batch, N, F)
            reward = decode_rewards[:, t + 1]    # (batch, 1) - reward at t+1
            
            # Predict next observation
            pred_next_obs = self.decode_obs(latent, current_obs, current_action)
            step_obs_loss = F.mse_loss(pred_next_obs, next_obs)
            recon_obs_loss += step_obs_loss
            obs_losses_per_step.append(step_obs_loss.item())
            
            # Predict reward  
            pred_reward = self.decode_reward(latent, current_obs, current_action, next_obs)
            step_reward_loss = F.mse_loss(pred_reward, reward)
            recon_reward_loss += step_reward_loss
            reward_losses_per_step.append(step_reward_loss.item())
        
        # Average over sequence length (avoid division by zero)
        if decode_seq_len > 1:
            recon_obs_loss /= (decode_seq_len - 1)
            recon_reward_loss /= (decode_seq_len - 1)
        else:
            # If sequence too short, set losses to zero
            recon_obs_loss = torch.tensor(0.0, device=obs_seq.device)
            recon_reward_loss = torch.tensor(0.0, device=obs_seq.device)
        
        # KL divergence with standard normal prior
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # Total loss
        total_loss = recon_obs_loss + recon_reward_loss + beta * kl_loss
        
        # Logging
        if self.training:
            self.training_step += 1
        
        # === NEW: Enhanced loss components with detailed breakdowns ===
        loss_components = {
            # === Original components ===
            'recon_obs': recon_obs_loss.item(),
            'recon_reward': recon_reward_loss.item(),
            'kl': kl_loss.item(),
            'total': total_loss.item(),
            
            # === NEW: Detailed breakdowns ===
            'recon_obs_per_step': float(recon_obs_loss.item() / max(decode_seq_len - 1, 1)),
            'recon_reward_per_step': float(recon_reward_loss.item() / max(decode_seq_len - 1, 1)),
            'kl_per_batch': float(kl_loss.item() / batch_size),
            'sequence_length': int(decode_seq_len),
            'context_length': int(context_len) if context_len else int(decode_seq_len),
            'encode_length': int(encode_obs.shape[1]),
            
            # === NEW: Latent statistics ===
            'latent_mu_mean': float(mu.mean().item()),
            'latent_mu_std': float(mu.std().item()),
            'latent_mu_max': float(mu.max().item()),
            'latent_mu_min': float(mu.min().item()),
            'latent_logvar_mean': float(logvar.mean().item()),
            'latent_logvar_std': float(logvar.std().item()),
            'latent_logvar_max': float(logvar.max().item()),
            'latent_logvar_min': float(logvar.min().item()),
            'latent_dim': int(self.latent_dim),
            
            # === NEW: Latent distribution properties ===
            'latent_kl_per_dim': float(kl_loss.item() / self.latent_dim),
            'latent_effective_dim': float(torch.sum(torch.exp(logvar) > 0.1).item()),  # Dims with meaningful variance
            'latent_posterior_norm': float(torch.norm(mu, dim=1).mean().item()),
            'latent_prior_kl_divergence': float(kl_loss.item()),
            
            # === NEW: Reconstruction quality metrics ===
            'obs_reconstruction_mse': float(recon_obs_loss.item()),
            'reward_reconstruction_mse': float(recon_reward_loss.item()),
            'obs_reconstruction_rmse': float(torch.sqrt(recon_obs_loss).item()),
            'reward_reconstruction_rmse': float(torch.sqrt(recon_reward_loss).item()),
            'reconstruction_ratio': float(recon_obs_loss.item() / max(recon_reward_loss.item(), 1e-8)),
            
            # === NEW: Per-timestep statistics ===
            'obs_loss_per_step_mean': float(np.mean(obs_losses_per_step)) if obs_losses_per_step else 0.0,
            'obs_loss_per_step_std': float(np.std(obs_losses_per_step)) if len(obs_losses_per_step) > 1 else 0.0,
            'obs_loss_per_step_max': float(np.max(obs_losses_per_step)) if obs_losses_per_step else 0.0,
            'reward_loss_per_step_mean': float(np.mean(reward_losses_per_step)) if reward_losses_per_step else 0.0,
            'reward_loss_per_step_std': float(np.std(reward_losses_per_step)) if len(reward_losses_per_step) > 1 else 0.0,
            'reward_loss_per_step_max': float(np.max(reward_losses_per_step)) if reward_losses_per_step else 0.0,
            
            # === NEW: Training progress and hyperparameters ===
            'training_step': int(self.training_step),
            'beta_weight': float(beta),
            'beta_weighted_kl': float(beta * kl_loss.item()),
            'unweighted_total_loss': float((recon_obs_loss + recon_reward_loss + kl_loss).item()),
            'kl_weight_ratio': float(beta * kl_loss.item() / max(total_loss.item(), 1e-8)),
            
            # === NEW: Batch and architecture info ===
            'batch_size': int(batch_size),
            'num_assets': int(obs_seq.shape[2]) if len(obs_seq.shape) > 2 else 0,
            'num_features': int(obs_seq.shape[3]) if len(obs_seq.shape) > 3 else 0,
            'hidden_dim': int(self.hidden_dim),
            
            # === NEW: Loss component ratios ===
            'obs_loss_fraction': float(recon_obs_loss.item() / max(total_loss.item(), 1e-8)),
            'reward_loss_fraction': float(recon_reward_loss.item() / max(total_loss.item(), 1e-8)),
            'kl_loss_fraction': float(beta * kl_loss.item() / max(total_loss.item(), 1e-8)),
            
            # === NEW: Gradient information (if available) ===
            'has_gradients': bool(mu.grad is not None),
            'mu_grad_norm': float(torch.norm(mu.grad).item()) if mu.grad is not None else 0.0,
            'logvar_grad_norm': float(torch.norm(logvar.grad).item()) if logvar.grad is not None else 0.0,
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
        