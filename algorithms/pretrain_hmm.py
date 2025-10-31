import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from hmmlearn.hmm import GaussianHMM
from copy import deepcopy
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from environments.data import PortfolioDataset
from models.hmm_encoder import HMMEncoder
from config import experiment_to_training_config, ExperimentConfig

logger = logging.getLogger(__name__)


class HMMPretrainingLogger:
    """
    Lightweight CSV logger for recording metrics and parameters during HMM pretraining.
    Each run produces a timestamped CSV file under `experiment_logs/`.
    """

    def __init__(self, asset_class: str, seed: int):
        self.asset_class = asset_class
        self.seed = seed
        self.output_dir = Path("experiment_logs")
        self.output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"hmm_pretraining_{asset_class}_seed{seed}_{timestamp}.csv"
        self.metrics = []

        logger.info(f"HMM pretraining logger initialized at {self.csv_path}")

    def log_metric(self, metric_name: str, value: float, step: int = 0):
        """Record a single scalar metric (e.g., loss, likelihood)."""
        self.metrics.append({
            "timestamp": datetime.now().isoformat(),
            "asset_class": self.asset_class,
            "seed": self.seed,
            "step": step,
            "metric": metric_name,
            "value": value,
        })

    def log_params(self, params: dict):
        """Record configuration parameters (logged with step = -1)."""
        for key, value in params.items():
            if isinstance(value, (int, float, bool)):
                self.metrics.append({
                    "timestamp": datetime.now().isoformat(),
                    "asset_class": self.asset_class,
                    "seed": self.seed,
                    "step": -1,
                    "metric": f"param_{key}",
                    "value": float(value),
                })

    def save_metrics(self):
        """Write all recorded metrics to disk."""
        if not self.metrics:
            return
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.csv_path, index=False)
        logger.info(f"HMM pretraining metrics saved to {self.csv_path}")


def pretrain_hmm(asset_class: str, seed: int = 0, config=None):
    """
    Perform offline HMM fitting and distill its latent regime structure into a neural encoder.

    Args:
        asset_class (str): Dataset to use ("sp500" or "crypto").
        seed (int): Random seed for reproducibility.
        config: Optional pre-configured `TrainingConfig`. If None, one is constructed.

    Returns:
        tuple[bool, str | None]:
            - success (bool): True if pretraining completed successfully.
            - encoder_path (str | None): Path to the saved encoder checkpoint, or None on failure.
    """
    csv_logger = HMMPretrainingLogger(asset_class, seed)

    try:
        # Prepare configuration
        if config is None:
            exp = ExperimentConfig(seed=seed, asset_class=asset_class, encoder="hmm")
            cfg = experiment_to_training_config(exp)
        else:
            cfg = config

        cfg.latent_dim = 4  # fixed number of HMM states
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        logger.info(f"Starting HMM pretraining for {asset_class} (seed={seed})")

        csv_logger.log_params({
            "asset_class": asset_class,
            "seed": seed,
            "hmm_states": cfg.latent_dim,
            "hidden_dim": cfg.hidden_dim,
            "device": str(device),
        })

        # Dataset loading
        if cfg.asset_class == "crypto":
            cfg.proportional = True
            cfg.proportions = (0.7, 0.2, 0.1)
            logger.info(f"Using proportional splits for crypto dataset: {cfg.proportions}")

        full_dataset = PortfolioDataset(
            asset_class=cfg.asset_class,
            data_path=cfg.data_path,
            train_end=cfg.train_end,
            val_end=cfg.val_end,
            proportional=getattr(cfg, "proportional", False),
            proportions=getattr(cfg, "proportions", (0.7, 0.2, 0.1)),
        )

        train_split = full_dataset.get_split("train")

        # Extract and reshape features for HMM fitting
        features = train_split.data[train_split.feature_cols].values.reshape(
            len(train_split),
            train_split.num_assets,
            train_split.num_features,
        )
        X = features.reshape(-1, train_split.num_features)
        logger.info(f"HMM training data shape: {X.shape}")
        csv_logger.log_metric("training_samples", X.shape[0])

        # Fit Gaussian HMM
        logger.info(f"Fitting GaussianHMM with {cfg.latent_dim} states...")
        hmm = GaussianHMM(
            n_components=cfg.latent_dim,
            covariance_type="full",
            n_iter=100,
            random_state=seed,
            tol=1e-3,
        )
        hmm.fit(X)

        converged = hmm.monitor_.converged
        log_likelihood = hmm.score(X)
        if not converged:
            logger.warning("HMM did not converge fully; proceeding with current parameters")

        _, posteriors = hmm.score_samples(X)
        posteriors = posteriors.reshape(len(train_split), train_split.num_assets, cfg.latent_dim)

        csv_logger.log_metric("hmm_converged", int(converged))
        csv_logger.log_metric("hmm_log_likelihood", log_likelihood)

        # Build distillation dataset
        obs_tensor = torch.tensor(features, dtype=torch.float32)
        actions = torch.zeros((len(train_split), train_split.num_assets))
        rewards = torch.zeros((len(train_split), 1))
        targets = torch.tensor(posteriors.mean(axis=1), dtype=torch.float32)

        full_ds = TensorDataset(obs_tensor, actions, rewards, targets)
        val_size = int(len(full_ds) * 0.1)
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(
            full_ds,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        # Initialize encoder
        encoder = HMMEncoder(
            obs_dim=(train_split.num_assets, train_split.num_features),
            num_assets=train_split.num_assets,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(device)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

        # Train encoder
        best_loss = float("inf")
        best_model_state = None
        patience = 5
        wait = 0
        max_epochs = 50

        logger.info("Beginning encoder distillation training")

        for epoch in range(max_epochs):
            encoder.train()
            total_loss = 0.0
            for obs, act, rew, target in train_loader:
                obs, act, rew, target = (
                    obs.to(device),
                    act.to(device),
                    rew.to(device),
                    target.to(device),
                )
                regime_probs = encoder.encode(obs.unsqueeze(1), act.unsqueeze(1), rew.unsqueeze(1))
                loss = F.kl_div(regime_probs.log(), target, reduction="batchmean")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Validation
            encoder.eval()
            val_loss = 0.0
            with torch.no_grad():
                for obs, act, rew, target in val_loader:
                    obs, act, rew, target = (
                        obs.to(device),
                        act.to(device),
                        rew.to(device),
                        target.to(device),
                    )
                    regime_probs = encoder.encode(obs.unsqueeze(1), act.unsqueeze(1), rew.unsqueeze(1))
                    loss = F.kl_div(regime_probs.log(), target, reduction="batchmean")
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)

            logger.info(f"Epoch {epoch + 1:02d}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")
            csv_logger.log_metric("train_loss", avg_train_loss, step=epoch + 1)
            csv_logger.log_metric("val_loss", avg_val_loss, step=epoch + 1)

            # Early stopping
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model_state = deepcopy(encoder.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        if best_model_state is not None:
            encoder.load_state_dict(best_model_state)
            logger.info(f"Restored best encoder (val_loss={best_loss:.4f})")

        # Save encoder checkpoint
        checkpoints_dir = Path("checkpoints")
        checkpoints_dir.mkdir(exist_ok=True)
        encoder_path = checkpoints_dir / f"hmm_{asset_class}_seed{seed}_encoder.pt"
        torch.save(encoder.state_dict(), encoder_path)
        logger.info(f"Saved pretrained encoder to {encoder_path}")

        # Final metrics
        csv_logger.log_metric("final_train_loss", avg_train_loss)
        csv_logger.log_metric("best_val_loss", best_loss)
        csv_logger.log_metric("training_epochs", epoch + 1)
        csv_logger.log_metric("success", 1.0)
        csv_logger.save_metrics()

        logger.info(f"HMM pretraining completed successfully for {asset_class}")
        return True, str(encoder_path)

    except Exception as e:
        logger.error(f"HMM pretraining failed: {e}")
        csv_logger.log_metric("success", 0.0)
        csv_logger.log_metric("error", hash(str(e)) % 1_000_000)
        csv_logger.save_metrics()
        return False, None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Pretrain HMM encoder for portfolio optimization.")
    parser.add_argument("--asset_class", type=str, required=True, choices=["sp500", "crypto"],
                        help="Dataset to train the HMM on.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    args = parser.parse_args()

    success, encoder_path = pretrain_hmm(asset_class=args.asset_class, seed=args.seed)
    if success:
        logger.info(f"Encoder saved to {encoder_path}")
    else:
        logger.error("HMM pretraining failed.")
        exit(1)
