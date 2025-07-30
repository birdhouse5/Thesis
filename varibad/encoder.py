"""
VAE Encoder for Portfolio VariBAD
Processes market trajectories to infer task embeddings (market regimes)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

class PortfolioEncoder(nn.Module):
    """
    RNN-based encoder that processes market trajectories to produce
    posterior distribution over market regime embeddings.
    """
    
    def __init__(
        self,
        input_dim: int,  # Market features dimension
        hidden_dim: int,
        latent_dim: int,  # Task embedding dimension
        num_layers: int
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Input embedding layer
        self.input_embed = nn.Linear(input_dim, hidden_dim)
        
        # RNN to process sequential market data
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layers for mean and log-variance
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(
        self, 
        trajectory: torch.Tensor,  # [batch_size, seq_len, input_dim]
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            trajectory: Market trajectory [batch, seq_len, features]
            hidden: Previous hidden state (optional)
            
        Returns:
            mu: Mean of posterior [batch, latent_dim]
            logvar: Log-variance of posterior [batch, latent_dim]  
            hidden: Final hidden state
        """
        batch_size, seq_len, _ = trajectory.shape
        
        # Embed input features
        embedded = torch.relu(self.input_embed(trajectory))
        
        # Process through RNN
        rnn_out, hidden = self.rnn(embedded, hidden)
        
        # Use final timestep output
        final_output = rnn_out[:, -1, :]  # [batch, hidden_dim]
        
        # Compute posterior parameters
        mu = self.fc_mu(final_output)
        logvar = self.fc_logvar(final_output)
        
        return mu, logvar, hidden
    
    def encode(self, trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience method for encoding without hidden state."""
        mu, logvar, _ = self.forward(trajectory)
        return mu, logvar
    
    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from posterior using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


# Test the encoder
if __name__ == "__main__":
    # Example usage
    batch_size, seq_len, input_dim = 32, 50, 20
    latent_dim = 8
    
    encoder = PortfolioEncoder(
        input_dim=input_dim,
        hidden_dim=64,
        latent_dim=latent_dim,
        num_layers=2
    )
    
    # Mock market trajectory
    trajectory = torch.randn(batch_size, seq_len, input_dim)
    
    # Encode
    mu, logvar, hidden = encoder(trajectory)
    
    print(f"Input shape: {trajectory.shape}")
    print(f"Posterior mu shape: {mu.shape}")
    print(f"Posterior logvar shape: {logvar.shape}")
    print(f"Hidden state shape: {hidden.shape}")
    
    # Sample from posterior
    z = encoder.sample(mu, logvar)
    print(f"Sampled embedding shape: {z.shape}")