import torch
from environments.env import MetaEnv, normalize_with_budget_constraint
from environments.data import PortfolioDataset

# Load dataset
dataset = PortfolioDataset(
    asset_class="sp500",
    data_path="environments/data/sp500_rl_ready_cleaned.parquet",
    force_recreate=False
)

datasets = dataset.get_all_splits()
train_split = datasets['train']

# Get a LARGER window (need more than seq_len for sampling)
window = train_split.get_window_tensor(0, 400, device='cpu')  # Changed from 200 to 400

# Create environment
env = MetaEnv(
    dataset={'features': window['features'], 'raw_prices': window['raw_prices']},
    feature_columns=train_split.feature_cols,
    seq_len=200,
    min_horizon=150,
    max_horizon=200,
)

# Set task
task = env.sample_task()
env.set_task(task)
obs = env.reset()

# Test with raw action (unnormalized)
raw_action = torch.randn(30) * 2.0  # Random values [-2, 2]

print("Testing normalization:")
print(f"Raw action range: [{raw_action.min():.3f}, {raw_action.max():.3f}]")
print(f"Raw action sum(|a|): {torch.abs(raw_action).sum():.3f}")

# Step through environment
next_obs, reward, done, info = env.step(raw_action)

# Check returned weights
returned_weights = info['weights']
print(f"\nReturned weights range: [{returned_weights.min():.3f}, {returned_weights.max():.3f}]")
print(f"Returned weights sum(|w|): {torch.abs(returned_weights).sum():.3f}")
print(f"Cash position: {info['cash_pct']:.3f}")
print(f"Budget check (sum|w| + cash): {torch.abs(returned_weights).sum() + info['cash_pct']:.3f}")
print(f"✅ Budget constraint satisfied!" if abs(torch.abs(returned_weights).sum() + info['cash_pct'] - 1.0) < 0.01 else "❌ Budget constraint VIOLATED!")

# Test direct reward computation (backtest path)
env.current_step = 1
reward2, weights2, w_cash2, turnover, cost, _, _ = env.compute_reward_with_capital(raw_action)

print(f"\nDirect reward computation:")
print(f"Weights range: [{weights2.min():.3f}, {weights2.max():.3f}]")
print(f"Weights sum(|w|): {torch.abs(weights2).sum():.3f}")
print(f"Cash: {w_cash2:.3f}")
print(f"Budget check (sum|w| + cash): {torch.abs(weights2).sum() + w_cash2:.3f}")
print(f"✅ Budget constraint satisfied!" if abs(torch.abs(weights2).sum() + w_cash2 - 1.0) < 0.01 else "❌ Budget constraint VIOLATED!")

# Verify both paths give same weights
weights_match = torch.allclose(returned_weights, weights2, atol=1e-5)
print(f"\n✅ Both paths produce identical weights!" if weights_match else "❌ Weights differ between paths!")

print(f"\nReward comparison:")
print(f"  env.step() reward: {reward:.6f}")
print(f"  compute_reward_with_capital() reward: {reward2:.6f}")