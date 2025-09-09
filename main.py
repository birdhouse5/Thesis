import mlflow
from pathlib import Path
from datetime import datetime
import torch

# --- import your config system ---
from config import (
    generate_experiment_configs,
    experiment_to_training_config,
    TrainingConfig
)

# --- import your env + models + trainer ---
from environments.data_preparation import create_dataset
from environments.env import MetaEnv
from models.policy import PortfolioPolicy
from models.vae import VAE
from algorithms.trainer import PPOTrainer

# --- experiment runner ---
def run_training(cfg: TrainingConfig):
    """Run one training job end-to-end (no MLflow logging yet)."""

    # === 1. Ensure dataset exists ===
    if not Path(cfg.data_path).exists():
        print(f"Dataset not found at {cfg.data_path}, creating...")
        cfg.data_path = create_dataset(cfg.data_path)

    # === 2. Setup environment ===
    from environments.dataset import create_split_datasets

    datasets = create_split_datasets(
        data_path=cfg.data_path,
        train_end=cfg.train_end,
        val_end=cfg.val_end
    )

    # Convert datasets → tensors (like in old ExperimentRunner)
    split_tensors = {}
    for split_name, dataset in datasets.items():
        features_list, prices_list = [], []
        num_windows = max(1, (len(dataset) - cfg.seq_len) // cfg.seq_len)

        for i in range(num_windows):
            start, end = i * cfg.seq_len, (i+1) * cfg.seq_len
            if end <= len(dataset):
                window = dataset.get_window(start, end)
                features_list.append(torch.tensor(window['features'], dtype=torch.float32))
                prices_list.append(torch.tensor(window['raw_prices'], dtype=torch.float32))

        all_features = torch.stack(features_list)
        all_prices = torch.stack(prices_list)

        split_tensors[split_name] = {
            'features': all_features.view(-1, cfg.num_assets, dataset.num_features),
            'raw_prices': all_prices.view(-1, cfg.num_assets),
            'feature_columns': dataset.feature_cols,
            'num_windows': len(features_list)
        }

    # Create environments
    environments = {}
    for split_name, tensor_data in split_tensors.items():
        environments[split_name] = MetaEnv(
            dataset={
                'features': tensor_data['features'],
                'raw_prices': tensor_data['raw_prices']
            },
            feature_columns=tensor_data['feature_columns'],
            seq_len=cfg.seq_len,
            min_horizon=cfg.min_horizon,
            max_horizon=cfg.max_horizon
        )

    train_env, val_env = environments['train'], environments['val']

    # === 3. Build models ===
    obs_shape = train_env.reset().shape
    device = torch.device(cfg.device)

    vae = None
    if cfg.encoder == "vae":
        vae = VAE(
            obs_dim=obs_shape,
            num_assets=cfg.num_assets,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim
        ).to(device)

    policy = PortfolioPolicy(
        obs_shape=obs_shape,
        latent_dim=cfg.latent_dim,
        num_assets=cfg.num_assets,
        hidden_dim=cfg.hidden_dim
    ).to(device)

    # === 4. Trainer ===
    trainer = PPOTrainer(env=train_env, policy=policy, vae=vae, config=cfg)

    # === 5. Training loop ===
    episodes_trained = 0
    while episodes_trained < cfg.max_episodes:
        result = trainer.train_episode()   # your trainer already supports this
        episodes_trained += 1

        if episodes_trained % cfg.val_interval == 0:
            val_result = evaluate(val_env, policy, vae, cfg)
            print(f"Episode {episodes_trained} | Val Sharpe: {val_result['avg_reward']:.4f}")

    print(f"Training complete: {episodes_trained} episodes")
    return {"episodes_trained": episodes_trained}



def main():
    # === setup MLflow tracking ===
    mlflow.set_experiment("full_study_and_ablation")

    # === generate configs ===
    exps = generate_experiment_configs(num_seeds=10)

    # === run all experiments ===
    for exp in exps:
        cfg = experiment_to_training_config(exp)
        print(f"[{datetime.now()}] Starting run {cfg.exp_name}")
        run_training(cfg)


if __name__ == "__main__":
    main()
