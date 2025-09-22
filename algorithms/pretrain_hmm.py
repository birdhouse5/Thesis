import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from hmmlearn.hmm import GaussianHMM
from copy import deepcopy
import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime

from environments.data import PortfolioDataset, DatasetSplit
from models.hmm_encoder import HMMEncoder
from config import experiment_to_training_config, ExperimentConfig

logger = logging.getLogger(__name__)


class HMMPretrainingLogger:
    """Simple CSV logger for HMM pre-training metrics."""
    
    def __init__(self, asset_class: str, seed: int):
        self.asset_class = asset_class
        self.seed = seed
        self.output_dir = Path("experiment_logs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create filename with timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"hmm_pretraining_{asset_class}_seed{seed}_{timestamp}.csv"
        self.metrics = []
        
        logger.info(f"HMM pre-training logger initialized: {self.csv_path}")
    
    def log_metric(self, metric_name: str, value: float, step: int = 0):
        """Log a single metric."""
        self.metrics.append({
            'timestamp': datetime.now().isoformat(),
            'asset_class': self.asset_class,
            'seed': self.seed,
            'step': step,
            'metric': metric_name,
            'value': value
        })
    
    def log_params(self, params: dict):
        """Log parameters as metrics with step -1."""
        for key, value in params.items():
            if isinstance(value, (int, float, bool)):
                self.metrics.append({
                    'timestamp': datetime.now().isoformat(),
                    'asset_class': self.asset_class,
                    'seed': self.seed,
                    'step': -1,  # Use -1 to indicate parameters
                    'metric': f"param_{key}",
                    'value': float(value)
                })
    
    def save_metrics(self):
        """Save all logged metrics to CSV."""
        if self.metrics:
            df = pd.DataFrame(self.metrics)
            df.to_csv(self.csv_path, index=False)
            logger.info(f"HMM pre-training metrics saved to {self.csv_path}")


def pretrain_hmm(asset_class: str, seed: int = 0, config=None):
    """
    Fit offline HMM and distill into HMMEncoder, with CSV logging.
    
    Args:
        asset_class: "sp500" or "crypto"
        seed: Random seed
        config: Optional TrainingConfig object for parameter access
        
    Returns:
        tuple: (success: bool, encoder_path: str or None)
    """
    
    # Initialize CSV logger
    csv_logger = HMMPretrainingLogger(asset_class, seed)
    
    try:
        # Build config with HMM-specific settings
        if config is None:
            exp = ExperimentConfig(seed=seed, asset_class=asset_class, encoder="hmm")
            cfg = experiment_to_training_config(exp)
        else:
            cfg = config
        
        # Ensure HMM uses 4 states regardless of VAE latent_dim
        cfg.latent_dim = 4
        
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        logger.info(f"Starting HMM pretraining for {asset_class} with seed {seed}")
        
        # Log parameters
        csv_logger.log_params({
            'asset_class': asset_class,
            'seed': seed,
            'hmm_states': cfg.latent_dim,
            'hidden_dim': cfg.hidden_dim,
            'device': str(device)
        })
        
        # --- Handle data splits consistently with main pipeline ---
        if cfg.asset_class == "crypto":
            # Use proportional splitting for crypto
            cfg.proportional = True
            cfg.proportions = (0.7, 0.2, 0.1)
            logger.info(f"Using proportional splits for crypto: {cfg.proportions}")
                    
        # Load full dataset and get training split
        full_dataset = PortfolioDataset(
            asset_class=cfg.asset_class,
            data_path=cfg.data_path,
            train_end=cfg.train_end,
            val_end=cfg.val_end,
            proportional=getattr(cfg, 'proportional', False),
            proportions=getattr(cfg, 'proportions', (0.7, 0.2, 0.1))
        )
        
        # Get the training split
        train_split = full_dataset.get_split("train")
        
        # Get features in same format as main pipeline
        features = train_split.data[train_split.feature_cols].values.reshape(
            len(train_split), train_split.num_assets, train_split.num_features
        )
        
        # Flatten for HMM training
        X = features.reshape(-1, train_split.num_features)
        logger.info(f"HMM training data shape: {X.shape}")
        csv_logger.log_metric("training_samples", X.shape[0])

        # --- Fit Gaussian HMM with error handling ---
        logger.info(f"Fitting HMM with {cfg.latent_dim} states...")
        hmm = GaussianHMM(
            n_components=cfg.latent_dim, 
            covariance_type="full", 
            n_iter=100, 
            random_state=seed,
            tol=1e-3
        )
        
        hmm.fit(X)
        
        # Check convergence
        converged = hmm.monitor_.converged
        log_likelihood = hmm.score(X)
        
        if not converged:
            logger.warning("HMM did not converge, but proceeding with current parameters")
        
        # Get posteriors
        _, posteriors = hmm.score_samples(X)
        posteriors = posteriors.reshape(len(train_split), train_split.num_assets, cfg.latent_dim)
        
        logger.info("HMM fitting completed successfully")
        csv_logger.log_metric("hmm_converged", int(converged))
        csv_logger.log_metric("hmm_log_likelihood", log_likelihood)

        # --- Prepare PyTorch dataset for distillation ---
        obs_tensor = torch.tensor(features, dtype=torch.float32)
        actions = torch.zeros((len(train_split), train_split.num_assets))  # placeholder
        rewards = torch.zeros((len(train_split), 1))  # placeholder
        targets = torch.tensor(posteriors.mean(axis=1), dtype=torch.float32)  # avg regime probs

        full_ds = TensorDataset(obs_tensor, actions, rewards, targets)

        # Train/val split for encoder training
        val_split = 0.1
        val_size = int(len(full_ds) * val_split)
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size], 
                                      generator=torch.Generator().manual_seed(seed))

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        # --- Initialize encoder with correct dimensions ---
        encoder = HMMEncoder(
            obs_dim=(train_split.num_assets, train_split.num_features),
            num_assets=train_split.num_assets,
            latent_dim=cfg.latent_dim,  # 4 states
            hidden_dim=cfg.hidden_dim
        ).to(device)
        
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

        # --- Train encoder with early stopping ---
        best_loss = float("inf")
        best_model_state = None
        patience = 5
        wait = 0
        max_epochs = 50

        logger.info("Starting encoder distillation training...")
        
        for epoch in range(max_epochs):
            # Training
            encoder.train()
            total_loss = 0
            for obs, act, rew, target in train_loader:
                obs, act, rew, target = obs.to(device), act.to(device), rew.to(device), target.to(device)
                
                regime_probs = encoder.encode(obs.unsqueeze(1), act.unsqueeze(1), rew.unsqueeze(1))
                loss = F.kl_div(regime_probs.log(), target, reduction="batchmean")
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)

            # Validation
            encoder.eval()
            val_loss = 0
            with torch.no_grad():
                for obs, act, rew, target in val_loader:
                    obs, act, rew, target = obs.to(device), act.to(device), rew.to(device), target.to(device)
                    regime_probs = encoder.encode(obs.unsqueeze(1), act.unsqueeze(1), rew.unsqueeze(1))
                    loss = F.kl_div(regime_probs.log(), target, reduction="batchmean")
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)

            logger.info(f"Epoch {epoch+1:2d}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
            
            # Log metrics
            csv_logger.log_metric("train_loss", avg_train_loss, step=epoch+1)
            csv_logger.log_metric("val_loss", avg_val_loss, step=epoch+1)

            # Early stopping check
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model_state = deepcopy(encoder.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Restore best model
        if best_model_state is not None:
            encoder.load_state_dict(best_model_state)
            logger.info(f"Restored best model with validation loss: {best_loss:.4f}")

        # --- Save encoder to checkpoints directory ---
        checkpoints_dir = Path("checkpoints")
        checkpoints_dir.mkdir(exist_ok=True)
        encoder_path = checkpoints_dir / f"hmm_{asset_class}_seed{seed}_encoder.pt"
        
        torch.save(encoder.state_dict(), encoder_path)
        logger.info(f"💾 Saved pre-trained HMM encoder to {encoder_path}")
        
        # Log final metrics
        csv_logger.log_metric("final_train_loss", avg_train_loss)
        csv_logger.log_metric("best_val_loss", best_loss)
        csv_logger.log_metric("training_epochs", epoch + 1)
        csv_logger.log_metric("success", 1.0)
        
        # Save all metrics
        csv_logger.save_metrics()
        
        logger.info(f"✅ HMM pretraining completed successfully for {asset_class}")
        return True, str(encoder_path)

    except Exception as e:
        logger.error(f"HMM pretraining failed: {str(e)}")
        csv_logger.log_metric("success", 0.0)
        csv_logger.log_metric("error", hash(str(e)) % 1000000)  # Simple error hash
        csv_logger.save_metrics()
        return False, None


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Pretrain HMM encoder for portfolio optimization")
    parser.add_argument("--asset_class", type=str, required=True, choices=["sp500", "crypto"],
                       help="Asset class to train HMM on")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    success, encoder_path = pretrain_hmm(asset_class=args.asset_class, seed=args.seed)
    
    if success:
        logger.info(f"HMM pretraining completed successfully, encoder saved to {encoder_path}")
    else:
        logger.error("HMM pretraining failed")
        exit(1)