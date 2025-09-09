# smoke_test.py
import torch
import logging
from pathlib import Path

# Import your existing modules
from environments import data_preparation as dp
from environments.dataset import create_split_datasets
from environments.env import MetaEnv
from models.vae import VAE
from models.policy import PortfolioPolicy
from algorithms.trainer import PPOTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data_if_needed(data_path: Path, days: int = 10):
    """
    Create the crypto parquet if it does not exist.
    """
    if data_path.exists():
        logger.info(f"✅ Dataset already exists: {data_path}")
        return

    logger.info("🚀 Downloading crypto OHLCV data...")
    df = dp.sample_crypto(dp.CRYPTO_TICKERS, days=days, interval="15m")

    logger.info("✨ Adding technical indicators...")
    df = dp.add_technical_indicators(df)

    logger.info("🔄 Normalizing features...")
    df = dp.normalize_features(df)

    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(data_path, index=False)
    logger.info(f"💾 Saved cleaned dataset: {data_path} (shape={df.shape})")


def run_smoke_test(data_path="environments/data/crypto_dataset.parquet"):
    """
    Minimal smoke test for the full VariBAD PPO pipeline.
    Uses tiny dimensions and very few episodes.
    """
    data_path = Path(data_path)
    prepare_data_if_needed(data_path)

    # --- Minimal config ---
    class Config:
        # Model
        latent_dim = 8
        hidden_dim = 16
        policy_lr = 1e-3
        vae_lr = 1e-3
        vae_beta = 0.1
        vae_update_freq = 1
        vae_batch_size = 4
        batch_size = 8

        # Environment
        seq_len = 20
        min_horizon = 5
        max_horizon = 5
        num_assets = 5     # only use 5 assets for speed
        device = "cpu"

        # PPO
        ppo_epochs = 1
        discount_factor = 0.99
        gae_lambda = 0.95
        ppo_clip_ratio = 0.2
        entropy_coef = 0.01
        value_loss_coef = 0.5
        max_grad_norm = 0.5

        # Training schedule
        num_envs = 1
        max_episodes = 2
        episodes_per_task = 1

    cfg = Config()

    # --- Load dataset (crypto) ---
    datasets = create_split_datasets(
        data_path=str(data_path),
        proportional=True,   # proportional split for crypto
    )
    train_ds = datasets["train"]

    # Restrict to first num_assets
    tickers = train_ds.tickers[:cfg.num_assets]
    train_ds.data = train_ds.data[train_ds.data["ticker"].isin(tickers)].copy()
    train_ds.tickers = tickers
    train_ds.num_assets = len(tickers)

    # --- Build environment tensors ---
    window = train_ds.get_window(0, cfg.seq_len)
    features = torch.tensor(window["features"], dtype=torch.float32)
    prices = torch.tensor(window["raw_prices"], dtype=torch.float32)

    dataset_tensors = {"features": features, "raw_prices": prices}

    env = MetaEnv(
        dataset=dataset_tensors,
        feature_columns=train_ds.feature_cols,
        seq_len=cfg.seq_len,
        min_horizon=cfg.min_horizon,
        max_horizon=cfg.max_horizon,
    )

    # --- Build models ---
    obs_shape = features.shape[1:]  # (N, F)
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

    logger.info("✅ Smoke test completed successfully.")


if __name__ == "__main__":
    run_smoke_test()
