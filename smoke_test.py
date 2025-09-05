import types
import numpy as np
import torch
import pytest

import importlib
import sys
from pathlib import Path

# Import the user's module
main = importlib.import_module("main")

# -------------------- Stubs --------------------

class DummyDataset:
    def __init__(self, seq_len: int, num_assets: int, num_features: int, num_windows: int = 2):
        # make length large enough to produce at least 1 window
        self._len = seq_len * num_windows
        self.seq_len = seq_len
        self.num_assets = num_assets
        self.num_features = num_features
        self.feature_cols = [f"f{i}" for i in range(num_features)]

    def __len__(self):
        return self._len

    def get_window(self, start_idx, end_idx):
        L = end_idx - start_idx
        feats = np.random.randn(L, self.num_assets, self.num_features).astype("float32")
        prices = np.random.randn(L, self.num_assets).astype("float32")
        return {"features": feats, "raw_prices": prices}

class DummyEnv:
    def __init__(self, dataset, feature_columns, seq_len, min_horizon, max_horizon):
        self.dataset = dataset
        self.feature_columns = feature_columns
        self.seq_len = seq_len
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self.t = 0
        self.horizon = 5
        self.initial_capital = 100.0
        self.current_capital = self.initial_capital
        # observation shape: (num_assets, num_features)
        self._obs_shape = (dataset["features"].shape[1], dataset["features"].shape[2])

    def sample_task(self):
        return {"dummy": True}

    def set_task(self, task):
        self.t = 0
        self.current_capital = self.initial_capital

    def reset(self):
        self.t = 0
        # Return a deterministic observation
        return np.zeros(self._obs_shape, dtype="float32")

    def step(self, action):
        # action is expected shape (num_assets,)
        self.t += 1
        reward = float(np.random.randn() * 0.01)
        self.current_capital += reward
        done = self.t >= self.horizon
        # next obs
        obs = np.zeros(self._obs_shape, dtype="float32")
        info = {}
        return obs, reward, done, info

class DummyVAE(torch.nn.Module):
    def __init__(self, obs_dim, num_assets, latent_dim, hidden_dim):
        super().__init__()
        self.latent_dim = latent_dim

    def to(self, device):
        return self

    def eval(self):
        return self

    def train(self):
        return self

    def state_dict(self):
        return {}

    def encode(self, obs_seq, action_seq, reward_seq):
        mu = torch.zeros((1, self.latent_dim))
        logvar = torch.zeros((1, self.latent_dim))
        return mu, logvar, None

    def reparameterize(self, mu, logvar):
        return torch.zeros_like(mu)

class DummyPolicy(torch.nn.Module):
    def __init__(self, obs_shape, latent_dim, num_assets, hidden_dim):
        super().__init__()
        self.num_assets = num_assets

    def to(self, device):
        return self

    def eval(self):
        return self

    def train(self):
        return self

    def state_dict(self):
        return {}

    def act(self, obs_tensor, latent, deterministic=True):
        # return a simple zero action and dummy value
        action = torch.zeros((1, self.num_assets), dtype=torch.float32)
        value = torch.tensor([0.0])
        return action, value

class DummyTrainer:
    def __init__(self, env, policy, vae, config):
        self.env = env
        self.policy = policy
        self.vae = vae
        self.config = config
        self._episodes = 0

    def train_episode(self):
        self._episodes += 1
        # Do nothing but pretend an episode happened
        return {}

def tiny_seed_configs():
    cfg = main.ValidationConfig(
        seed=0,
        exp_name="smoke",
        device="cpu",
        # keep tiny for speed
        latent_dim=8,
        hidden_dim=16,
        seq_len=8,
        episodes_per_task=1,
        batch_size=32,
        vae_batch_size=8,
        ppo_epochs=1,
        max_episodes=3,
        val_interval=1,
        val_episodes=2,
        num_assets=4,
        min_horizon=2,
        max_horizon=3,
    )
    return [cfg]

# -------------------- The smoke test --------------------

def test_validation_pipeline_smoke(monkeypatch, tmp_path):
    # Make sure the dataset path check returns True to skip data creation
    monkeypatch.setattr(Path, "exists", lambda self: True, raising=False)

    # Stub out heavy components
    monkeypatch.setattr(main, "create_split_datasets",
                        lambda data_path, train_end, val_end: {
                            "train": DummyDataset(seq_len=8, num_assets=4, num_features=3),
                            "val": DummyDataset(seq_len=8, num_assets=4, num_features=3),
                        })

    monkeypatch.setattr(main, "MetaEnv", DummyEnv)
    monkeypatch.setattr(main, "VAE", DummyVAE)
    monkeypatch.setattr(main, "PortfolioPolicy", DummyPolicy)
    monkeypatch.setattr(main, "PPOTrainer", DummyTrainer)
    monkeypatch.setattr(main, "seed_everything", lambda seed: None)
    monkeypatch.setattr(main, "create_seed_configs", tiny_seed_configs)

    # Use a temporary results directory
    results_dir = tmp_path / "validation_results"
    runner = main.ExperimentRunner(results_dir=str(results_dir))

    # Run the full main() orchestrator to exercise logging, analysis, saving, etc.
    stats, results = main.main()

    # Basic sanity checks
    assert isinstance(stats, dict)
    assert "iqm" in stats
    assert len(results) == 1
    assert getattr(results[0], "episodes_trained", 1) > 0

    # Verify output artifacts exist
    assert (results_dir / "detailed_results.csv").exists()
    assert (results_dir / "statistical_summary.json").exists()
    assert (results_dir / "performance_profile.png").exists()
