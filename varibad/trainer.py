"""
PyTorch Lightning Trainer for Portfolio VariBAD
Combines VAE (encoder/decoder) and policy training
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Tuple, Optional, Any
import numpy as np
from torch.distributions import Normal

from varibad.encoder import PortfolioEncoder
from varibad.decoder import PortfolioDecoder 
from varibad.models.policy import PortfolioPolicy


class PortfolioVariBAD(pl.LightningModule):
    """
    Lightning module combining VAE and Policy for portfolio optimization.
    """
    
    def __init__(
        self,
        state_dim: int,
        num_assets: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        sequence_length: int = 50,
        encoder_lr: float = 1e-3,
        decoder_lr: float = 1e-3,
        policy_lr: float = 3e-4,
        kl_weight: float = 0.1,
        transaction_cost: float = 0.001,
        use_cash: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model dimensions
        self.state_dim = state_dim
        self.num_assets = num_assets
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.action_dim = num_assets + (1 if use_cash else 0)
        
        # Training hyperparameters
        self.encoder_lr = encoder_lr
        self.decoder_lr = decoder_lr
        self.policy_lr = policy_lr
        self.kl_weight = kl_weight
        self.transaction_cost = transaction_cost
        
        # Models
        self.encoder = PortfolioEncoder(
            input_dim=state_dim + self.action_dim + 1,  # state + action + reward
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=1
        )
        
        self.decoder = PortfolioDecoder(
            state_dim=state_dim,
            action_dim=self.action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_assets=num_assets
        )
        
        self.policy = PortfolioPolicy(
            state_dim=state_dim,
            latent_dim=latent_dim,
            num_assets=num_assets,
            hidden_dim=hidden_dim,
            use_cash=use_cash
        )
        
        # Separate optimizers for VAE and Policy
        self.automatic_optimization = False
        
    def configure_optimizers(self):
        """Configure separate optimizers for encoder, decoder, and policy."""
        encoder_params = list(self.encoder.parameters())
        decoder_params = list(self.decoder.parameters())
        policy_params = list(self.policy.parameters())
        
        encoder_optimizer = torch.optim.Adam(encoder_params, lr=self.encoder_lr)
        decoder_optimizer = torch.optim.Adam(decoder_params, lr=self.decoder_lr)
        policy_optimizer = torch.optim.Adam(policy_params, lr=self.policy_lr)
        
        return [encoder_optimizer, decoder_optimizer, policy_optimizer]
    
    def forward(
        self, 
        trajectory: torch.Tensor,  # [batch, seq_len, state+action+reward]
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through VAE and policy.
        
        Args:
            trajectory: Market trajectory [batch, seq_len, features]
            deterministic: Whether to use deterministic policy
            
        Returns:
            Dictionary with model outputs
        """
        batch_size, seq_len, _ = trajectory.shape
        
        # Split trajectory into components
        states = trajectory[:, :, :self.state_dim]
        actions = trajectory[:, :, self.state_dim:self.state_dim + self.action_dim]
        rewards = trajectory[:, :, -1:]
        
        # Encode trajectory to get task posterior
        mu, logvar, _ = self.encoder(trajectory)
        
        # Sample task embedding
        z = self.encoder.sample(mu, logvar) if not deterministic else mu
        
        # Generate actions for each timestep using policy
        policy_actions = []
        policy_values = []
        
        for t in range(seq_len):
            current_state = states[:, t]
            prev_action = actions[:, t-1] if t > 0 else torch.zeros_like(actions[:, 0])
            
            action, value = self.policy(current_state, z, prev_action, deterministic)
            policy_actions.append(action)
            policy_values.append(value)
        
        policy_actions = torch.stack(policy_actions, dim=1)
        policy_values = torch.stack(policy_values, dim=1)
        
        return {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'policy_actions': policy_actions,
            'policy_values': policy_values,
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'trajectory': trajectory  # Include original trajectory
        }
    
    def compute_vae_loss(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute VAE reconstruction and KL losses."""
        mu = outputs['mu']
        logvar = outputs['logvar']
        z = outputs['z']
        states = outputs['states']
        actions = outputs['actions']
        rewards = outputs['rewards']
        
        batch_size, seq_len, _ = states.shape
        
        # Reconstruction losses
        recon_losses = []
        
        for t in range(seq_len - 1):  # Predict next timestep
            current_state = states[:, t]
            current_action = actions[:, t]
            next_state = states[:, t + 1]
            reward = rewards[:, t + 1]
            
            # Decode next state and reward
            pred_next_state, pred_reward, _ = self.decoder(current_state, current_action, z)
            
            # Reconstruction losses
            state_loss = F.mse_loss(pred_next_state, next_state, reduction='none').sum(dim=-1)
            reward_loss = F.mse_loss(pred_reward.squeeze(-1), reward.squeeze(-1), reduction='none')
            
            recon_losses.append(state_loss + reward_loss)
        
        # Average reconstruction loss
        recon_loss = torch.stack(recon_losses, dim=1).mean()
        
        # KL divergence (regularization)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        
        # Total VAE loss
        vae_loss = recon_loss + self.kl_weight * kl_loss
        
        return {
            'vae_loss': vae_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def compute_policy_loss(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute policy loss (simplified actor-critic)."""
        rewards = outputs['rewards']
        policy_values = outputs['policy_values']
        policy_actions = outputs['policy_actions']
        actions = outputs['actions']
        
        batch_size, seq_len, _ = rewards.shape
        
        # Compute returns (simple cumulative reward)
        returns = []
        cumulative_reward = torch.zeros(batch_size, device=self.device)
        
        for t in reversed(range(seq_len)):
            cumulative_reward = rewards[:, t, 0] + 0.99 * cumulative_reward  # γ = 0.99
            returns.insert(0, cumulative_reward)
        
        returns = torch.stack(returns, dim=1)
        
        # Value loss (MSE between predicted values and returns)
        value_loss = F.mse_loss(policy_values.squeeze(-1), returns)
        
        # Policy loss (advantage-weighted)
        advantages = returns - policy_values.squeeze(-1).detach()
        
        # Action log probability (simplified - treating as regression)
        action_loss = F.mse_loss(policy_actions, actions) * advantages.abs().mean()
        
        # Transaction cost penalty
        transaction_costs = []
        for t in range(1, seq_len):
            cost = torch.abs(policy_actions[:, t] - policy_actions[:, t-1]).sum(dim=-1)
            transaction_costs.append(cost)
        
        transaction_cost_penalty = torch.stack(transaction_costs, dim=1).mean() * self.transaction_cost
        
        # Total policy loss
        policy_loss = value_loss + action_loss + transaction_cost_penalty
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'action_loss': action_loss,
            'transaction_cost': transaction_cost_penalty,
            'avg_return': returns.mean()
        }
    
    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Training step with separate encoder, decoder, and policy optimization."""
        encoder_opt, decoder_opt, policy_opt = self.optimizers()
        
        # Extract trajectory tensor from batch (Lightning returns tuple/list)
        if isinstance(batch, (list, tuple)):
            trajectory = batch[0]
        else:
            trajectory = batch
        
        # Forward pass
        outputs = self.forward(trajectory)
        
        # Compute VAE loss
        vae_losses = self.compute_vae_loss(outputs)
        
        # Update encoder
        encoder_opt.zero_grad()
        self.manual_backward(vae_losses['kl_loss'], retain_graph=True)
        encoder_opt.step()
        
        # Update decoder  
        decoder_opt.zero_grad()
        self.manual_backward(vae_losses['recon_loss'])
        decoder_opt.step()
        
        # Compute policy loss (detach VAE components)
        with torch.no_grad():
            outputs_detached = self.forward(trajectory)  # Fresh forward pass
            
        policy_losses = self.compute_policy_loss(outputs_detached)
        
        # Update policy
        policy_opt.zero_grad()
        self.manual_backward(policy_losses['policy_loss'])
        policy_opt.step()
        
        # Logging
        self.log_dict({
            'train_encoder_lr': self.encoder_lr,
            'train_decoder_lr': self.decoder_lr,
            'train_policy_lr': self.policy_lr,
            'train_vae_loss': vae_losses['vae_loss'],
            'train_recon_loss': vae_losses['recon_loss'],
            'train_kl_loss': vae_losses['kl_loss'],
            'train_policy_loss': policy_losses['policy_loss'],
            'train_value_loss': policy_losses['value_loss'],
            'train_return': policy_losses['avg_return']
        })
        
        return vae_losses['vae_loss'] + policy_losses['policy_loss']
    
    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        # Extract trajectory tensor from batch (Lightning returns tuple/list)
        if isinstance(batch, (list, tuple)):
            trajectory = batch[0]
        else:
            trajectory = batch
            
        outputs = self.forward(trajectory, deterministic=True)
        vae_losses = self.compute_vae_loss(outputs)
        policy_losses = self.compute_policy_loss(outputs)
        
        self.log_dict({
            'val_vae_loss': vae_losses['vae_loss'],
            'val_policy_loss': policy_losses['policy_loss'],
            'val_return': policy_losses['avg_return']
        })
        
        return vae_losses['vae_loss'] + policy_losses['policy_loss']
    
    def predict_action(
        self,
        state: torch.Tensor,
        trajectory: torch.Tensor,
        prev_action: torch.Tensor,
        deterministic: bool = True
    ) -> torch.Tensor:
        """Predict portfolio action for inference."""
        self.eval()
        with torch.no_grad():
            # Encode trajectory to get task embedding
            mu, logvar, _ = self.encoder(trajectory)
            z = mu if deterministic else self.encoder.sample(mu, logvar)
            
            # Generate action
            action, _ = self.policy(state, z, prev_action, deterministic)
            
        return action
    
    def compute_losses_standalone(self, trajectory: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute losses without Lightning trainer (for testing).
        This method can be called directly without needing optimizers() method.
        """
        # Forward pass
        outputs = self.forward(trajectory)
        
        # Compute losses
        vae_losses = self.compute_vae_loss(outputs)
        policy_losses = self.compute_policy_loss(outputs)
        
        return {**vae_losses, **policy_losses}


# Example usage and testing
if __name__ == "__main__":
    # Model parameters
    state_dim = 50
    num_assets = 30
    latent_dim = 16
    sequence_length = 20
    batch_size = 32
    
    # Create model
    model = PortfolioVariBAD(
        state_dim=state_dim,
        num_assets=num_assets,
        latent_dim=latent_dim,
        sequence_length=sequence_length
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Mock batch data
    # trajectory format: [batch, seq_len, state_dim + action_dim + reward_dim]
    action_dim = num_assets + 1  # +1 for cash
    trajectory_dim = state_dim + action_dim + 1
    
    batch = torch.randn(batch_size, sequence_length, trajectory_dim)
    
    # Test forward pass
    outputs = model(batch)
    print(f"Forward pass successful!")
    print(f"Task embedding shape: {outputs['z'].shape}")
    print(f"Policy actions shape: {outputs['policy_actions'].shape}")
    
    # Test loss computation using standalone method (no trainer needed)
    model.train()
    
    # Use the standalone loss computation method
    all_losses = model.compute_losses_standalone(batch)
    
    print(f"VAE loss: {all_losses['vae_loss']:.4f}")
    print(f"Reconstruction loss: {all_losses['recon_loss']:.4f}")
    print(f"KL loss: {all_losses['kl_loss']:.4f}")
    print(f"Policy loss: {all_losses['policy_loss']:.4f}")
    print(f"Average return: {all_losses['avg_return']:.4f}")
    
    print("\nModel ready for Lightning Trainer!")
    
    # Test with actual Lightning trainer
    print("\nTesting with Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='cpu',
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        max_steps=2  # Just test a few steps
    )
    
    # Create a simple dataset
    from torch.utils.data import TensorDataset, DataLoader
    
    # Generate some dummy data - each sample should be a full trajectory
    num_samples = 100
    dummy_trajectories = torch.randn(num_samples, sequence_length, trajectory_dim)
    dataset = TensorDataset(dummy_trajectories)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Test training step with real trainer
    try:
        trainer.fit(model, dataloader)
        print("✅ Lightning training successful!")
    except Exception as e:
        print(f"❌ Lightning training failed: {e}")
        import traceback
        traceback.print_exc()