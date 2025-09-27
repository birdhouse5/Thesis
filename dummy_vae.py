import torch
from models.vae import VAE

# Create VAE
vae = VAE(obs_dim=(30, 35), num_assets=30, latent_dim=32, hidden_dim=256)

# Create dummy data
obs = torch.randn(2, 100, 30, 35)
actions = torch.randn(2, 100, 30)
rewards = torch.randn(2, 100, 1)

# Test 5 times - context_fraction should vary
print("Testing asymmetric encoding:")
for i in range(5):
    loss, info = vae.compute_loss(obs, actions, rewards, beta=0.1)
    print(f"  Run {i}: context_fraction={info['context_fraction']:.2f}, context_len={info['context_len']}")

# Expected output: context_fraction should vary between 0.5 and 1.0