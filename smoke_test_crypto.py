# smoke_test_crypto.py
import torch
from pathlib import Path

from environments.dataset import create_split_datasets
from environments.env import MetaEnv
from models.vae import VAE
from models.policy import PortfolioPolicy
from algorithms.trainer import PPOTrainer

def smoke_test_crypto(data_path="environments/data/crypto_rl_ready_cleaned.parquet"):
    # 1. Load dataset splits (proportional)
    datasets = create_split_datasets(
        data_path=data_path,
        proportional=True,
        proportions=(0.7, 0.2, 0.1)
    )

    # 2. Build environment on train split
    train_ds = datasets["train"]
    env = MetaEnv(
        dataset=train_ds.get_window(0, len(train_ds)),  # full window
        feature_columns=train_ds.feature_cols,
        seq_len=60,
        min_horizon=10,
        max_horizon=10
    )


    # 3. Sample one task to infer obs shape
    task = env.sample_task()
    env.set_task(task)
    obs0 = env.reset()
    obs_shape = obs0.shape  # (num_assets, num_features)
    num_assets = obs_shape[0]

    # 4. Build models
    latent_dim = 16
    hidden_dim = 64
    device = "cpu"

    vae = VAE(obs_dim=obs_shape, num_assets=num_assets,
              latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    policy = PortfolioPolicy(obs_shape=obs_shape, latent_dim=latent_dim,
                             num_assets=num_assets, hidden_dim=hidden_dim).to(device)

    # 5. Minimal trainer config (dataclass-like)
    class DummyConfig:
        policy_lr = 1e-3
        vae_lr = 1e-3
        batch_size = 16
        vae_batch_size = 16
        min_horizon = 10
        max_horizon = 10
        discount_factor = 0.99
        gae_lambda = 0.95
        ppo_epochs = 1
        ppo_clip_ratio = 0.2
        value_loss_coef = 0.5
        entropy_coef = 0.01
        latent_dim = latent_dim
        num_assets = num_assets
        device = device
        disable_vae = False
        vae_update_freq = 1

    config = DummyConfig()
    trainer = PPOTrainer(env, policy, vae, config)

    # 6. Run one training episode
    print("=== Running crypto smoke test ===")
    result = trainer.train_episode()
    print("Result:", result)

if __name__ == "__main__":
    smoke_test_crypto()
