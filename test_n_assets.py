# test_n_assets.py
from config import ExperimentConfig, experiment_to_training_config
from main import prepare_environments

# Test with 5 assets
exp = ExperimentConfig(seed=0, asset_class="sp500", encoder="vae", n_assets=5)
cfg = experiment_to_training_config(exp)

print(f"Config n_assets_limit: {cfg.n_assets_limit}")

envs, split_tensors, datasets = prepare_environments(cfg)

print(f"Train env num_assets: {datasets['train'].num_assets}")
print(f"Tickers: {datasets['train'].tickers}")
print(f"Expected: 5 assets")