import torch
import torch.nn as nn
import torch.nn.functional as F


class HMMEncoder(nn.Module):
    """
    Stub HMM encoder that outputs 4-dimensional regime probabilities.
    Placeholder for future HMM implementation.
    """
    def __init__(self, obs_dim, num_assets, reward_dim=1, latent_dim=4, hidden_dim=256):
        super(HMMEncoder, self).__init__()
        
        self.obs_dim = obs_dim          # (N, F)
        self.action_dim = num_assets    # N
        self.reward_dim = reward_dim    # 1
        self.latent_dim = latent_dim    # 4 (regime probabilities)
        self.hidden_dim = hidden_dim
        
        # Simple neural network that maps inputs to regime probabilities
        obs_flat_dim = obs_dim[0] * obs_dim[1]  # N × F
        self.obs_encoder = nn.Linear(obs_flat_dim, 64)
        self.action_encoder = nn.Linear(num_assets, 32)
        self.reward_encoder = nn.Linear(reward_dim, 16)
        
        # Combine features and output regime probabilities
        combined_dim = 64 + 32 + 16
        self.regime_predictor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),  # 4 regime probabilities
            nn.Softmax(dim=-1)  # Ensure probabilities sum to 1
        )
        
    def encode(self, obs_sequence, action_sequence, reward_sequence):
        """
        Encode trajectory to regime probabilities.
        
        Args:
            obs_sequence: (batch, seq_len, N, F)
            action_sequence: (batch, seq_len, N)
            reward_sequence: (batch, seq_len, 1)
            
        Returns:
            regime_probs: (batch, 4) - probabilities over 4 regimes
        """
        batch_size, seq_len = obs_sequence.shape[:2]
        
        # Use most recent observation for regime detection
        current_obs = obs_sequence[:, -1]  # (batch, N, F)
        current_action = action_sequence[:, -1]  # (batch, N)  
        current_reward = reward_sequence[:, -1]  # (batch, 1)
        
        # Encode inputs
        obs_flat = current_obs.reshape(batch_size, -1)
        obs_emb = F.relu(self.obs_encoder(obs_flat))
        action_emb = F.relu(self.action_encoder(current_action))
        reward_emb = F.relu(self.reward_encoder(current_reward))
        
        # Combine and predict regime
        combined = torch.cat([obs_emb, action_emb, reward_emb], dim=-1)
        regime_probs = self.regime_predictor(combined)
        
        return regime_probs
    
    def reparameterize(self, regime_probs, logvar=None):
        """
        For compatibility with VAE interface.
        Just return the regime probabilities.
        """
        return regime_probs
    
    def compute_loss(self, obs_seq, action_seq, reward_seq, beta=0.1, context_len=None):
        """
        Stub loss function. HMM training happens offline.
        Return zero loss during training.
        """
        device = obs_seq.device
        return torch.tensor(0.0, device=device), {"hmm_loss": 0.0}