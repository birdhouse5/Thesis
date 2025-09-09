# smoke_test_sp500.py
import torch
import logging
from pathlib import Path

from environments.dataset import create_split_datasets
from environments.env import MetaEnv
from environments import data_preparation as dp
from models.vae import VAE
from models.policy import PortfolioPolicy
from algorithms.trainer import PPOTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_sp500_if_needed(data_path: Path):
    """
    Ensure the S&P500 parquet exists, otherwise build it.
    """
    if data_path.exists():
        logger.info(f"✅ Dataset already exists: {data_path}")
        return

    logger.info("📈 Preparing S&P500 dataset...")
    # Use your repo's function to build the cleaned parquet
    dp.create_dataset(str(data_path))
    logger.info(f"✅ Created S&P500 dataset: {data_path}")

def run_smoke_test(data_path="environments/data/sp500_rl_ready_cleaned.parquet"):
    """
    Minimal smoke test for the full VariBAD PPO pipeline on S&P500 dataset.
    Uses tiny dimensions and very few episodes.
    """
    data_path = Path(data_path)
    prepare_sp500_if_needed(data_path)

    # --- Minimal config ---
    class Config:
        latent_dim = 8
        hidden_dim = 16
        policy_lr = 1e-3
        vae_lr = 1e-3
        vae_beta = 0.1
        vae_update_freq = 1
        vae_batch_size = 4
        batch_size = 8

        seq_len = 20
        min_horizon = 5
        max_horizon = 5
        num_assets = 5   # keep only 5 tickers for speed
        device = "cpu"

        ppo_epochs = 1
        discount_factor = 0.99
        gae_lambda = 0.95
        ppo_clip_ratio = 0.2
        entropy_coef = 0.01
        value_loss_coef = 0.5
        max_grad_norm = 0.5

        num_envs = 1
        max_episodes = 2
        episodes_per_task = 1

    cfg = Config()

    # --- Load dataset ---
    datasets = create_split_datasets(
        data_path=str(data_path),
        proportional=False,   # date-based split for S&P500
    )
    train_ds = datasets["train"]

    # Restrict to subset of assets
    tickers = train_ds.tickers[:cfg.num_assets]
    train_ds.data = train_ds.data[train_ds.data["ticker"].isin(tickers)].copy()
    train_ds.tickers = tickers
    train_ds.num_assets = len(tickers)

    # --- Build environment tensors ---
    all_features = torch.tensor(
        train_ds.data[train_ds.feature_cols].values.reshape(
            train_ds.num_days, train_ds.num_assets, train_ds.num_features
        ),
        dtype=torch.float32,
    )
    all_prices = torch.tensor(
        train_ds.data["close"].values.reshape(
            train_ds.num_days, train_ds.num_assets
        ),
        dtype=torch.float32,
    )

    dataset_tensors = {"features": all_features, "raw_prices": all_prices}

    env = MetaEnv(
        dataset=dataset_tensors,
        feature_columns=train_ds.feature_cols,
        seq_len=cfg.seq_len,
        min_horizon=cfg.min_horizon,
        max_horizon=cfg.max_horizon,
    )

    # --- Build models ---
    obs_shape = all_features.shape[1:]
    vae = VAE(
        obs_dim=obs_shape,
        num_assets=cfg.num_assets,
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(cfg.device)
    policy = PortfolioPolicy(
        obs_shape=obs_shape,
        latent_dim=cfg.latent_dim,
        num_assets=cfg.num_assets,
        hidden_dim=cfg.hidden_dim,
    ).to(cfg.device)

    trainer = PPOTrainer(env=env, policy=policy, vae=vae, config=cfg)

    # --- Run a couple episodes ---
    for ep in range(cfg.max_episodes):
        env.set_task(env.sample_task())
        result = trainer.train_episode()
        logger.info(f"Episode {ep}: {result}")

    logger.info("✅ S&P500 smoke test completed successfully.")

if __name__ == "__main__":
    run_smoke_test()
