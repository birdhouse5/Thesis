import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import mlflow
import mlflow.pytorch
from hmmlearn.hmm import GaussianHMM
from copy import deepcopy
import logging


from environments.data import PortfolioDataset, DatasetSplit
from models.hmm_encoder import HMMEncoder
from config import experiment_to_training_config, ExperimentConfig
from mlflow_logger import setup_mlflow

logger = logging.getLogger(__name__)

def pretrain_hmm(asset_class: str, seed: int = 0):
    """Fit offline HMM and distill into HMMEncoder, then log to MinIO via MLflow."""
    
    # Build config with HMM-specific settings
    exp = ExperimentConfig(seed=seed, asset_class=asset_class, encoder="hmm")
    cfg = experiment_to_training_config(exp)
    
    # Ensure HMM uses 4 states regardless of VAE latent_dim
    cfg.latent_dim = 4
    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Setup MLflow
    backend = setup_mlflow()
    mlflow.set_experiment("hmm_pretraining")

    run_name = f"{asset_class}_hmm_pretraining_seed{seed}"
    
    with mlflow.start_run(run_name=run_name):
        logger.info(f"Starting HMM pretraining for {asset_class} with seed {seed}")
        mlflow.log_param("asset_class", asset_class)
        mlflow.log_param("seed", seed)
        mlflow.log_param("hmm_states", cfg.latent_dim)
        
        try:
            # --- Handle data splits consistently with main pipeline ---
            if cfg.asset_class == "crypto":
                # Use proportional splitting for crypto
                cfg.proportional = True
                cfg.proportions = (0.7, 0.2, 0.1)  # or your chosen fractions
                logger.info(f"Using proportional splits for crypto: {cfg.proportions}")
                        
            # Load training dataset only (HMM pretraining uses train split)
            dataset = PortfolioDataset(
                asset_class=cfg.asset_class,
                data_path=cfg.data_path,
                split="train",
                train_end=cfg.train_end,
                val_end=cfg.val_end,
                proportional=cfg.proportional,
                proportions=cfg.proportions
            )
            
            # Get features in same format as main pipeline
            features = dataset.data[dataset.feature_cols].values.reshape(
                len(dataset), dataset.num_assets, dataset.num_features
            )
            
            # Flatten for HMM training
            X = features.reshape(-1, dataset.num_features)
            logger.info(f"HMM training data shape: {X.shape}")

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
            if not hmm.monitor_.converged:
                logger.warning("HMM did not converge, but proceeding with current parameters")
            
            # Get posteriors
            _, posteriors = hmm.score_samples(X)
            posteriors = posteriors.reshape(len(dataset), dataset.num_assets, cfg.latent_dim)
            
            logger.info("HMM fitting completed successfully")
            mlflow.log_metric("hmm_converged", int(hmm.monitor_.converged))
            mlflow.log_metric("hmm_log_likelihood", hmm.score(X))

            # --- Prepare PyTorch dataset for distillation ---
            obs_tensor = torch.tensor(features, dtype=torch.float32)
            actions = torch.zeros((len(dataset), dataset.num_assets))  # placeholder
            rewards = torch.zeros((len(dataset), 1))  # placeholder
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
                obs_dim=(dataset.num_assets, dataset.num_features),
                num_assets=dataset.num_assets,
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
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch+1)
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch+1)

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

            # --- Save to MinIO via MLflow ---
            model_name = f"{asset_class}_hmm_encoder"
            
            mlflow.pytorch.log_model(
                encoder,
                artifact_path="encoder_model",
                registered_model_name=model_name
            )
            
            # Log final metrics
            mlflow.log_param("final_train_loss", avg_train_loss)
            mlflow.log_param("best_val_loss", best_loss)
            mlflow.log_param("training_epochs", epoch + 1)
            
            logger.info(f"✅ Pretrained HMM encoder saved as {model_name} in {backend} backend")
            return True

        except Exception as e:
            logger.error(f"HMM pretraining failed: {str(e)}")
            mlflow.log_param("error", str(e))
            return False


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Pretrain HMM encoder for portfolio optimization")
    parser.add_argument("--asset_class", type=str, required=True, choices=["sp500", "crypto"],
                       help="Asset class to train HMM on")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    success = pretrain_hmm(asset_class=args.asset_class, seed=args.seed)
    
    if success:
        logger.info("HMM pretraining completed successfully")
    else:
        logger.error("HMM pretraining failed")
        exit(1)