"""Test that all components are properly set up."""

import yaml
import torch
from src.models.varibad_trader import VariBADTrader
from src.data.data_loader import DataLoader

print("Testing setup...")

# Load config
try:
    with open('configs/varibad_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("✅ Config loaded successfully")
except Exception as e:
    print(f"❌ Config error: {e}")
    exit(1)

# Test data loader
try:
    loader = DataLoader(
        assets=config['data']['assets'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        train_end_date=config['data']['train_end_date']
    )
    sampler = loader.get_episode_sampler('train', config['episodes']['length'])
    episode = sampler.sample_episode()
    print(f"✅ Data loader works - episode shape: {episode.shape}")
except Exception as e:
    print(f"❌ Data loader error: {e}")
    exit(1)

# Test model creation
try:
    model = VariBADTrader(config)
    print("✅ Model created successfully")
    
    # Test forward pass with dummy data
    batch_size = 1
    state_dim = 50
    dummy_state = torch.randn(batch_size, state_dim)
    print(f"   Testing with state shape: {dummy_state.shape}")
    
    weights, posterior = model(dummy_state)
    print(f"✅ Forward pass works!")
    print(f"   Portfolio weights shape: {weights.shape}")
    print(f"   Portfolio weights: {weights.detach().numpy().squeeze()}")
    print(f"   Weights sum to: {weights.sum().item():.4f}")
    print(f"   Posterior mean shape: {posterior.mean.shape}")
    print(f"   Posterior std shape: {posterior.stddev.shape}")
except Exception as e:
    print(f"❌ Model error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n✅ All components working! Ready to implement training.")