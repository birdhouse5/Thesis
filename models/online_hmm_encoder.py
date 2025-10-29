import numpy as np
from hmmlearn.hmm import GaussianHMM
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class OnlineHMMEncoder(nn.Module):
    """
    HMM-based encoder using hmmlearn (replaces the PyTorch neural encoder).

    This version maintains the same interface but uses a true HMM backend.
    """

    def __init__(self, obs_dim, num_assets, reward_dim=1, latent_dim=4, hidden_dim=256):
        super(OnlineHMMEncoder, self).__init__()

        self.obs_dim = obs_dim
        self.num_assets = num_assets
        self.reward_dim = reward_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Flattened observation dimensionality
        self.obs_flat_dim = obs_dim[0] * obs_dim[1] + num_assets + reward_dim

        # Underlying HMM from hmmlearn
        # Use diagonal covariance for simplicity
        self.hmm = GaussianHMM(
            n_components=latent_dim,
            covariance_type="diag",
            n_iter=10,
            verbose=False,
            tol=1e-3
        )

        # Keep a small buffer of fitted flag
        self._is_fitted = False

        logger.info(f"Created OnlineHMMEncoder (hmmlearn-based) with {latent_dim} regimes")

    def _flatten_inputs(self, obs_seq, action_seq, reward_seq):
        """
        Combine obs, actions, rewards into flat features for HMM input.
        """
        obs_flat = obs_seq.reshape(obs_seq.shape[0], obs_seq.shape[1], -1)
        action_flat = action_seq.reshape(action_seq.shape[0], action_seq.shape[1], -1)
        reward_flat = reward_seq.reshape(reward_seq.shape[0], reward_seq.shape[1], -1)
        return torch.cat([obs_flat, action_flat, reward_flat], dim=-1)

    def encode(self, obs_seq, action_seq, reward_seq, return_logits=False):
        """
        Encode trajectory using HMM posterior probabilities.

        Returns:
            regime_probs: (batch, latent_dim)
        """
        with torch.no_grad():
            features = self._flatten_inputs(obs_seq, action_seq, reward_seq)
            batch_size, seq_len, feat_dim = features.shape

            regime_probs_batch = []

            for b in range(batch_size):
                X = features[b].cpu().numpy()

                # Ensure we have fitted the model at least once
                if not self._is_fitted:
                    # Initialize with random small dataset
                    init_data = np.random.randn(max(10, seq_len), feat_dim)
                    self.hmm.fit(init_data)
                    self._is_fitted = True

                try:
                    logprob, posteriors = self.hmm.score_samples(X)
                except Exception:
                    # In case of singular covariance or numerical issue
                    posteriors = np.ones((seq_len, self.latent_dim)) / self.latent_dim

                # Take last timestep posterior as regime representation
                regime_probs_batch.append(posteriors[-1])

            regime_probs = torch.tensor(np.array(regime_probs_batch), dtype=torch.float32)
            if return_logits:
                return torch.log(regime_probs + 1e-10)
            return regime_probs

    def reparameterize(self, regime_probs, logvar=None):
        """
        Keep API compatibility (returns probs or sample if you wish).
        """
        return regime_probs

    def partial_fit(self, obs_seq, action_seq, reward_seq):
        """
        Incrementally update HMM with new data (simulating online learning).
        """
        features = self._flatten_inputs(obs_seq, action_seq, reward_seq)
        X = features.reshape(-1, features.shape[-1]).detach().cpu().numpy()
        self.hmm.fit(X)
        self._is_fitted = True

    def compute_loss(
        self,
        obs_seq, action_seq, reward_seq,
        beta=0.1,
        num_elbo_terms=8,
        prior_mu=None, prior_logvar=None
    ):
        """
        Compute pseudo-loss: negative log-likelihood under HMM.
        """
        features = self._flatten_inputs(obs_seq, action_seq, reward_seq)
        batch_size, seq_len, feat_dim = features.shape

        total_nll = 0.0

        for b in range(batch_size):
            X = features[b].cpu().numpy()
            if not self._is_fitted:
                self.partial_fit(obs_seq, action_seq, reward_seq)
            try:
                nll = -self.hmm.score(X)
            except Exception:
                nll = 0.0
            total_nll += nll

        total_nll /= batch_size
        total_loss = torch.tensor(total_nll, dtype=torch.float32)

        return total_loss, {
            "total": total_loss.item(),
            "recon": 0.0,
            "consistency": 0.0,
            "entropy": 0.0,
            "num_elbo_terms": num_elbo_terms,
            "regime_confidence": 1.0,
        }


class GumbelHMMEncoder(OnlineHMMEncoder):
    """
    Dummy subclass to keep compatibility (same as OnlineHMMEncoder for hmmlearn)
    """
    def __init__(self, *args, temperature=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature


def create_online_hmm_encoder(obs_dim, num_assets, latent_dim=4, hidden_dim=256, use_gumbel=False):
    """
    Factory function (kept identical).
    """
    if use_gumbel:
        return GumbelHMMEncoder(
            obs_dim=obs_dim,
            num_assets=num_assets,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
    else:
        return OnlineHMMEncoder(
            obs_dim=obs_dim,
            num_assets=num_assets,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
