import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hmmlearn.hmm import GaussianHMM
import logging

logger = logging.getLogger(__name__)


class OnlineHMMEncoder(nn.Module):
    """
    Online HMM encoder that learns during RL training.
    Maintains a real HMM that gets refitted on recent experience,
    plus a neural network for fast inference.
    """
    def __init__(self, obs_dim, num_assets, reward_dim=1, latent_dim=4, hidden_dim=256):
        super(OnlineHMMEncoder, self).__init__()
        
        self.obs_dim = obs_dim          # (N, F)
        self.action_dim = num_assets    # N
        self.reward_dim = reward_dim    # 1
        self.latent_dim = latent_dim    # 4 (regime probabilities)
        self.hidden_dim = hidden_dim
        self.num_assets = num_assets
        
        # Neural network for fast inference (will be trained to match HMM)
        obs_flat_dim = obs_dim[0] * obs_dim[1]  # N × F
        self.obs_encoder = nn.Linear(obs_flat_dim, 64)
        self.action_encoder = nn.Linear(num_assets, 32)
        self.reward_encoder = nn.Linear(reward_dim, 16)
        
        # Combine features and output regime probabilities
        combined_dim = 64 + 32 + 16
        self.regime_predictor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.Softmax(dim=-1)
        )
        
        # The actual HMM (will be fitted online)
        self.hmm = None
        self.hmm_fitted = False
        self.refit_counter = 0
        
        logger.info(f"Initialized OnlineHMMEncoder with {latent_dim} regimes")
    
    def _init_hmm(self, n_features):
        """Initialize a new HMM with random parameters."""
        self.hmm = GaussianHMM(
            n_components=self.latent_dim,
            covariance_type="full",
            n_iter=50,  # Fewer iterations for online learning
            random_state=None,  # Will vary each refit
            tol=1e-3
        )
        logger.info(f"Initialized new GaussianHMM with {self.latent_dim} components")
    
    def refit_hmm(self, trajectory_buffer):
        """
        Refit the HMM on recent trajectories.
        
        Args:
            trajectory_buffer: List of trajectory dicts with 'observations', 'actions', 'rewards'
        
        Returns:
            success: bool indicating if refit was successful
            hmm_log_likelihood: float, HMM score on the data
        """
        if len(trajectory_buffer) < 5:
            logger.debug("Not enough trajectories to refit HMM")
            return False, 0.0
        
        try:
            # Collect features from trajectories
            all_features = []
            for traj in trajectory_buffer:
                # Get observations from trajectory
                obs = traj['observations']  # (T, N, F)
                
                # Flatten to (T*N, F) for HMM
                if torch.is_tensor(obs):
                    obs_np = obs.cpu().numpy()
                else:
                    obs_np = obs
                
                T = obs_np.shape[0]
                features_flat = obs_np.reshape(T * self.num_assets, -1)
                all_features.append(features_flat)
            
            # Concatenate all features
            X = np.vstack(all_features)
            
            # Initialize HMM if first time
            if self.hmm is None:
                self._init_hmm(n_features=X.shape[1])
            
            # Fit HMM
            logger.debug(f"Refitting HMM on {X.shape[0]} samples from {len(trajectory_buffer)} trajectories")
            self.hmm.fit(X)
            
            # Check convergence
            log_likelihood = self.hmm.score(X)
            converged = self.hmm.monitor_.converged
            
            if not converged:
                logger.warning(f"HMM did not converge (log_likelihood={log_likelihood:.2f})")
            else:
                logger.info(f"HMM refitted successfully (log_likelihood={log_likelihood:.2f})")
            
            self.hmm_fitted = True
            self.refit_counter += 1
            
            return True, log_likelihood
            
        except Exception as e:
            logger.error(f"HMM refit failed: {e}")
            return False, 0.0
    
    def get_hmm_posteriors(self, obs_sequence):
        """
        Get regime posteriors from the fitted HMM.
        
        Args:
            obs_sequence: (batch, seq_len, N, F)
        
        Returns:
            posteriors: (batch, latent_dim) - average regime probabilities
        """
        if not self.hmm_fitted:
            # Return uniform distribution if HMM not yet fitted
            batch_size = obs_sequence.shape[0]
            device = obs_sequence.device
            return torch.ones(batch_size, self.latent_dim, device=device) / self.latent_dim
        
        batch_size, seq_len, N, F = obs_sequence.shape
        
        # Convert to numpy
        if torch.is_tensor(obs_sequence):
            obs_np = obs_sequence.cpu().numpy()
        else:
            obs_np = obs_sequence
        
        # Get posteriors for each sequence
        batch_posteriors = []
        for b in range(batch_size):
            # Flatten sequence: (seq_len*N, F)
            seq_flat = obs_np[b].reshape(seq_len * N, -1)
            
            try:
                # Get posteriors from HMM
                _, posteriors = self.hmm.score_samples(seq_flat)
                
                # Reshape to (seq_len, N, n_components)
                posteriors_reshaped = posteriors.reshape(seq_len, N, self.latent_dim)
                
                # Average over time and assets to get task-level regime probs
                avg_posteriors = posteriors_reshaped.mean(axis=(0, 1))  # (n_components,)
                
                batch_posteriors.append(avg_posteriors)
                
            except Exception as e:
                logger.warning(f"Failed to compute HMM posteriors: {e}")
                # Fallback to uniform
                batch_posteriors.append(np.ones(self.latent_dim) / self.latent_dim)
        
        # Stack and convert to tensor
        posteriors_array = np.stack(batch_posteriors)
        device = obs_sequence.device if torch.is_tensor(obs_sequence) else 'cpu'
        return torch.tensor(posteriors_array, dtype=torch.float32, device=device)
    
    def encode(self, obs_sequence, action_sequence, reward_sequence):
        """
        Encode trajectory to regime probabilities using neural network.
        
        Args:
            obs_sequence: (batch, seq_len, N, F)
            action_sequence: (batch, seq_len, N)
            reward_sequence: (batch, seq_len, 1)
            
        Returns:
            regime_probs: (batch, latent_dim)
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
    
    def compute_distillation_loss(self, obs_seq, action_seq, reward_seq):
        """
        Compute loss for distilling HMM knowledge into neural network.
        
        Args:
            obs_seq: (batch, seq_len, N, F)
            action_seq: (batch, seq_len, N)
            reward_seq: (batch, seq_len, 1)
        
        Returns:
            loss: scalar tensor
            info: dict with loss components
        """
        if not self.hmm_fitted:
            # No loss if HMM not fitted yet
            device = obs_seq.device
            return torch.tensor(0.0, device=device), {
                "hmm_distillation_loss": 0.0,
                "hmm_fitted": False
            }
        
        # Get HMM targets (teacher)
        with torch.no_grad():
            hmm_targets = self.get_hmm_posteriors(obs_seq)
        
        # Get neural network predictions (student)
        neural_preds = self.encode(obs_seq, action_seq, reward_seq)
        
        # KL divergence loss (student should match teacher)
        loss = F.kl_div(
            neural_preds.log(),
            hmm_targets,
            reduction="batchmean"
        )
        
        return loss, {
            "hmm_distillation_loss": loss.item(),
            "hmm_fitted": True,
            "hmm_refit_count": self.refit_counter
        }
    
    def compute_loss(self, obs_seq, action_seq, reward_seq, beta=0.1, num_elbo_terms=None, 
                    prior_mu=None, prior_logvar=None):
        """
        Compute loss for online HMM learning.
        This is called during training like VAE.
        
        Returns distillation loss to match HMM posteriors.
        """
        return self.compute_distillation_loss(obs_seq, action_seq, reward_seq)
    
    def reparameterize(self, regime_probs, logvar=None):
        """
        For compatibility with VAE interface.
        Just return the regime probabilities.
        """
        return regime_probs