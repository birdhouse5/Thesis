# pretrain_hmm.py
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
from hmmlearn.hmm import GaussianHMM

from environments.dataset import Dataset
from models.hmm_encoder import HMMEncoder
from config import experiment_to_training_config, ExperimentConfig
from mlflow_setup import setup_mlflow


def pretrain_hmm(asset_class: str, seed: int = 0):
    """Fit offline HMM and distill into HMMEncoder, then log to MinIO via MLflow."""
    exp = ExperimentConfig(seed=seed, asset_class=asset_class, encoder="hmm")
    cfg = experiment_to_training_config(exp)
    device = torch.device(cfg.device)

    # Setup MLflow
    setup_mlflow()
    run_name = f"{asset_class}_hmm_pretraining"

    with mlflow.start_run(run_name=run_name):
        # --- Load dataset ---
        dataset = Dataset(cfg.data_path, split="train", train_end=cfg.train_end, val_end=cfg.val_end)
        features = dataset.data[dataset.feature_cols].values.reshape(len(dataset), dataset.num_assets, dataset.num_features)

        # Flatten for HMM training
        X = features.reshape(-1, dataset.num_features)

        # --- Fit Gaussian HMM ---
        hmm = GaussianHMM(n_components=cfg.latent_dim, covariance_type="full", n_iter=100, random_state=seed)
        hmm.fit(X)
        _, posteriors = hmm.score_samples(X)  # shape (timesteps, latent_dim)
        posteriors = posteriors.reshape(len(dataset), dataset.num_assets, cfg.latent_dim)

        # --- Prepare PyTorch dataset ---
        obs_tensor = torch.tensor(features, dtype=torch.float32)
        actions = torch.zeros((len(dataset), dataset.num_assets))  # placeholder
        rewards = torch.zeros((len(dataset), 1))                   # placeholder
        targets = torch.tensor(posteriors.mean(axis=1), dtype=torch.float32)  # avg regime probs

        train_ds = TensorDataset(obs_tensor, actions, rewards, targets)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

        # --- Init encoder ---
        encoder = HMMEncoder(obs_dim=(dataset.num_assets, dataset.num_features),
                             num_assets=dataset.num_assets,
                             latent_dim=cfg.latent_dim,
                             hidden_dim=cfg.hidden_dim).to(device)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

        # --- Distill HMM posteriors ---
        for epoch in range(10):
            total_loss = 0
            for obs, act, rew, target in train_loader:
                obs, act, rew, target = obs.to(device), act.to(device), rew.to(device), target.to(device)

                regime_probs = encoder.encode(obs.unsqueeze(1), act.unsqueeze(1), rew.unsqueeze(1))
                loss = F.kl_div(regime_probs.log(), target, reduction="batchmean")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss={total_loss/len(train_loader):.4f}")

        # --- Save to MinIO via MLflow ---
        model_name = f"{asset_class}_hmm_encoder"
        mlflow.pytorch.log_model(
            encoder,
            artifact_path="encoder_model",
            registered_model_name=model_name
        )
        mlflow.log_param("latent_dim", cfg.latent_dim)
        mlflow.log_param("asset_class", cfg.asset_class)
        print(f"✅ Pretrained HMM encoder stored in MinIO as {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_class", type=str, required=True, choices=["sp500", "crypto"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pretrain_hmm(asset_class=args.asset_class, seed=args.seed)
