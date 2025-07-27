"""
Fixed VariBAD VAE Models for Portfolio Optimization

Key fixes:
1. Corrected encoder input dimension calculation
2. Proper action dimension handling for long/short portfolios
3. Better tensor shape validation
4. Improved debugging and error messages

The main issue was the encoder expected different input dimensions
than what was being provided during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from torch.distributions import Normal


class TrajectoryEncoder(nn.Module):
    """
    Fixed Encoder q_φ(m|τ:t): Trajectory sequences → Belief parameters
    
    Input: τ:t = (s_0, a_0, r_1, s_1, a_1, r_2, ..., s_t)
    Output: Parameters (μ, σ) of posterior distribution q(m|τ:t)
    
    Key fix: Proper input dimension calculation
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int, 
                 latent_dim: int = 5,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 rnn_type: str = 'GRU'):
        """
        Args:
            state_dim: Portfolio state dimension 
            action_dim: Portfolio action dimension (includes long+short if applicable)
            latent_dim: Task embedding dimension
            hidden_dim: RNN hidden dimension
            num_layers: Number of RNN layers
            rnn_type: 'GRU' or 'LSTM'
        """
        super().__init__()
    
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # Input dimension: [state, action, reward] concatenated
        self.input_dim = state_dim + action_dim + 1  # +1 for reward
        
        # DEBUG: Print to verify correct calculation
        print(f"TrajectoryEncoder initialization:")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")  # Should be 60, not 30
        print(f"  Input dim: {self.input_dim}")  # Should be 1025
        
        # RNN to process trajectory sequences
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=self.input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.1 if num_layers > 1 else 0
            )
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.1 if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # Output layers for belief parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.xavier_uniform_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)
    
    def forward(self, 
                states: torch.Tensor,
                actions: torch.Tensor, 
                rewards: torch.Tensor,
                lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: trajectory sequence → belief parameters
        
        Args:
            states: [batch_size, max_seq_len, state_dim]
            actions: [batch_size, max_seq_len, action_dim]
            rewards: [batch_size, max_seq_len]
            lengths: [batch_size] - actual sequence lengths
            
        Returns:
            mu: [batch_size, latent_dim] - belief mean
            logvar: [batch_size, latent_dim] - belief log variance
        """
        batch_size, max_seq_len = states.shape[:2]
        
        # Debug: Print input shapes
        print(f"Encoder forward - Input shapes:")
        print(f"  States: {states.shape}")
        print(f"  Actions: {actions.shape}")
        print(f"  Rewards: {rewards.shape}")
        print(f"  Expected input dim: {self.input_dim}")
        
        # Validate input dimensions
        if states.shape[2] != self.state_dim:
            raise ValueError(f"State dimension mismatch: got {states.shape[2]}, expected {self.state_dim}")
        
        if actions.shape[2] != self.action_dim:
            raise ValueError(f"Action dimension mismatch: got {actions.shape[2]}, expected {self.action_dim}")
        
        # Concatenate inputs: [state, action, reward]
        rewards_expanded = rewards.unsqueeze(-1)  # [batch_size, max_seq_len, 1]
        trajectory_input = torch.cat([states, actions, rewards_expanded], dim=-1)
        
        print(f"  Concatenated input: {trajectory_input.shape}")
        print(f"  Expected: [batch_size, max_seq_len, {self.input_dim}]")
        
        # Verify concatenated input matches expected dimension
        if trajectory_input.shape[2] != self.input_dim:
            raise ValueError(
                f"Concatenated input dimension mismatch: "
                f"got {trajectory_input.shape[2]}, expected {self.input_dim}"
            )
        
        # Pack sequences for efficient RNN processing
        packed_input = nn.utils.rnn.pack_padded_sequence(
            trajectory_input, 
            lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # Process through RNN
        packed_output, hidden = self.rnn(packed_input)
        
        # Unpack output
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, 
            batch_first=True
        )
        
        # Use final hidden state for each sequence
        if isinstance(hidden, tuple):  # LSTM
            final_hidden = hidden[0][-1]  # Last layer hidden state
        else:  # GRU
            final_hidden = hidden[-1]  # Last layer hidden state
        
        # Compute belief parameters
        mu = self.fc_mu(final_hidden)
        logvar = self.fc_logvar(final_hidden)
        
        # Clamp log variance for numerical stability
        logvar = torch.clamp(logvar, -10, 10)
        
        return mu, logvar
    
    def sample_belief(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from belief distribution using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_belief_distribution(self, mu: torch.Tensor, logvar: torch.Tensor) -> Normal:
        """Get belief distribution for KL divergence computation."""
        std = torch.exp(0.5 * logvar)
        return Normal(mu, std)


class TrajectoryDecoder(nn.Module):
    """
    Decoder p_θ(τ_{t+1:H+}|m): Belief → Future predictions
    
    Input: Task embedding m + current state/action
    Output: Predictions for future states and rewards
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 latent_dim: int = 5,
                 hidden_dim: int = 128,
                 num_layers: int = 2):
        """
        Args:
            state_dim: Portfolio state dimension
            action_dim: Portfolio action dimension  
            latent_dim: Task embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # Input: [current_state, current_action, task_embedding]
        input_dim = state_dim + action_dim + latent_dim
        
        # Shared feature extractor
        layers = []
        prev_dim = input_dim
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Separate heads for different predictions
        self.state_predictor = nn.Linear(hidden_dim, state_dim)
        self.reward_predictor = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self,
                current_states: torch.Tensor,
                current_actions: torch.Tensor,
                task_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: current state/action + belief → future predictions
        """
        # Concatenate inputs
        decoder_input = torch.cat([current_states, current_actions, task_embeddings], dim=-1)
        
        # Process through shared layers
        features = self.shared_layers(decoder_input)
        
        # Generate predictions
        predicted_states = self.state_predictor(features)
        predicted_rewards = self.reward_predictor(features)
        
        return predicted_states, predicted_rewards


class VariBADPolicy(nn.Module):
    """
    Fixed Policy π_ψ(a_t|s_t, q(m|τ:t)): State + Belief → Portfolio weights
    
    Key fix: Proper action dimension handling for long/short portfolios
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,  # This should be the TOTAL action dimension from environment
                 latent_dim: int = 5,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 enable_short_selling: bool = True,
                 max_short_ratio: float = 0.3):
        """
        Args:
            state_dim: Portfolio state dimension
            action_dim: TOTAL action dimension from environment (e.g., 60 for 30 assets with long/short)
            latent_dim: Task embedding dimension 
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            enable_short_selling: Whether to allow short positions
            max_short_ratio: Maximum short position ratio
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim  # Total action dimension
        self.latent_dim = latent_dim
        self.enable_short_selling = enable_short_selling
        self.max_short_ratio = max_short_ratio
        
        # Infer number of assets from action dimension
        if enable_short_selling:
            # action_dim = 2 * num_assets (long + short weights)
            self.num_assets = action_dim // 2
            if action_dim != 2 * self.num_assets:
                raise ValueError(f"Action dim {action_dim} not divisible by 2 for long/short")
        else:
            # action_dim = num_assets (long weights only)
            self.num_assets = action_dim
        
        print(f"VariBADPolicy initialization:")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Num assets: {self.num_assets}")
        print(f"  Short selling: {enable_short_selling}")
        
        # Input: [state, belief_mu, belief_logvar]
        input_dim = state_dim + 2 * latent_dim  # μ and σ from encoder
        
        # Shared feature extractor
        layers = []
        prev_dim = input_dim
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Output layers for portfolio weights
        if enable_short_selling:
            # Output: [long_weights, short_weights]
            self.long_head = nn.Linear(hidden_dim, self.num_assets)
            self.short_head = nn.Linear(hidden_dim, self.num_assets)
        else:
            # Output: [long_weights] only
            self.long_head = nn.Linear(hidden_dim, self.num_assets)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self,
                states: torch.Tensor,
                belief_mu: torch.Tensor,
                belief_logvar: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: state + belief → portfolio weights
        
        Returns:
            portfolio_weights: [batch_size, action_dim] where action_dim matches environment
        """
        batch_size = states.shape[0]
        
        # Concatenate state with belief parameters
        belief_std = torch.exp(0.5 * belief_logvar)
        policy_input = torch.cat([states, belief_mu, belief_std], dim=-1)
        
        # Process through shared layers
        features = self.shared_layers(policy_input)
        
        # Generate portfolio weights
        long_logits = self.long_head(features)
        
        if self.enable_short_selling:
            short_logits = self.short_head(features)
            raw_weights = torch.cat([long_logits, short_logits], dim=-1)
        else:
            raw_weights = long_logits
        
        # Apply portfolio constraints
        portfolio_weights = self._apply_portfolio_constraints(raw_weights)
        
        # Verify output dimension matches expected action dimension
        assert portfolio_weights.shape[1] == self.action_dim, \
            f"Output dimension {portfolio_weights.shape[1]} != expected {self.action_dim}"
        
        return portfolio_weights
    
    def _apply_portfolio_constraints(self, raw_weights: torch.Tensor) -> torch.Tensor:
        """
        Apply MetaTrader portfolio constraints with proper dimension handling.
        """
        if self.enable_short_selling:
            batch_size = raw_weights.shape[0]
            
            # Split into long and short components
            long_logits = raw_weights[:, :self.num_assets]
            short_logits = raw_weights[:, self.num_assets:]
            
            # Apply softmax to long weights (ensures sum to 1)
            long_weights = F.softmax(long_logits, dim=-1)
            
            # Apply sigmoid to short weights, then scale and make negative
            short_probs = torch.sigmoid(short_logits)
            short_weights = -short_probs * self.max_short_ratio / self.num_assets
            
            # Simple conflict resolution: allow separate long/short positions
            # Only prevent extreme conflicts where positions are very similar
            conflict_threshold = 0.05
            
            long_strength = long_weights
            short_strength = torch.abs(short_weights)
            
            # Only resolve conflicts where positions are similar
            conflict_mask = torch.abs(long_strength - short_strength) < conflict_threshold
            
            # For conflicts, keep the stronger position
            keep_long_in_conflict = (long_strength >= short_strength) & conflict_mask
            keep_short_in_conflict = (short_strength > long_strength) & conflict_mask
            
            # Apply conflict resolution only where needed
            final_long = long_weights.clone()
            final_short = short_weights.clone()
            
            # Zero out weaker positions only in conflicts
            final_long = final_long * (~(conflict_mask & keep_short_in_conflict)).float()
            final_short = final_short * (~(conflict_mask & keep_long_in_conflict)).float()
            
            # Renormalize long weights to sum to 1 
            long_sum = final_long.sum(dim=-1, keepdim=True)
            final_long = final_long / (long_sum + 1e-8)
            
            # Combine final weights
            portfolio_weights = torch.cat([final_long, final_short], dim=-1)
            
        else:
            # Long-only case: simple softmax
            portfolio_weights = F.softmax(raw_weights, dim=-1)
        
        return portfolio_weights


class VariBADVAE(nn.Module):
    """
    Fixed Complete VariBAD VAE system with proper dimension handling.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,  # Total action dimension from environment
                 latent_dim: int = 5,
                 encoder_hidden: int = 128,
                 decoder_hidden: int = 128,
                 policy_hidden: int = 256,
                 enable_short_selling: bool = True,
                 max_short_ratio: float = 0.3):
        """
        Complete VariBAD system with corrected dimensions.
        
        Args:
            state_dim: Portfolio state dimension
            action_dim: TOTAL action dimension from environment (not just num_assets)
            latent_dim: Task embedding dimension
            enable_short_selling: Whether to allow short positions
            max_short_ratio: Maximum short position ratio
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim  # Store total action dimension
        self.latent_dim = latent_dim
        self.enable_short_selling = enable_short_selling
        
        print(f"VariBADVAE initialization:")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Short selling: {enable_short_selling}")
        
        # Initialize components with correct action dimension
        self.encoder = TrajectoryEncoder(
            state_dim=state_dim,
            action_dim=action_dim,  # Use total action dimension
            latent_dim=latent_dim,
            hidden_dim=encoder_hidden
        )
        
        self.decoder = TrajectoryDecoder(
            state_dim=state_dim,
            action_dim=action_dim,  # Use total action dimension
            latent_dim=latent_dim,
            hidden_dim=decoder_hidden
        )
        
        self.policy = VariBADPolicy(
            state_dim=state_dim,
            action_dim=action_dim,  # Use total action dimension
            latent_dim=latent_dim,
            hidden_dim=policy_hidden,
            enable_short_selling=enable_short_selling,
            max_short_ratio=max_short_ratio
        )
        
        # Prior distribution (standard normal)
        self.register_buffer('prior_mu', torch.zeros(latent_dim))
        self.register_buffer('prior_std', torch.ones(latent_dim))
        
        print(f"VariBADVAE initialized with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def encode_trajectory(self, 
                         states: torch.Tensor,
                         actions: torch.Tensor,
                         rewards: torch.Tensor,
                         lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode trajectory sequence to belief parameters and sample.
        """
        belief_mu, belief_logvar = self.encoder(states, actions, rewards, lengths)
        sampled_belief = self.encoder.sample_belief(belief_mu, belief_logvar)
        return belief_mu, belief_logvar, sampled_belief
    
    def get_policy_action(self,
                         current_state: torch.Tensor,
                         belief_mu: torch.Tensor,
                         belief_logvar: torch.Tensor) -> torch.Tensor:
        """Get portfolio action from policy given current state and belief."""
        return self.policy(current_state, belief_mu, belief_logvar)
    
    def compute_elbo(self,
                    states: torch.Tensor,
                    actions: torch.Tensor, 
                    rewards: torch.Tensor,
                    lengths: torch.Tensor,
                    next_states: torch.Tensor,
                    next_rewards: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute ELBO loss for VAE training.
        """
        batch_size = states.shape[0]
        
        # Encode trajectory to belief
        belief_mu, belief_logvar, sampled_belief = self.encode_trajectory(
            states, actions, rewards, lengths
        )
        
        # Decode to predict future
        # Use last state/action from each sequence
        last_indices = lengths - 1
        current_states = states[torch.arange(batch_size), last_indices]
        current_actions = actions[torch.arange(batch_size), last_indices]
        
        pred_states, pred_rewards = self.decoder(
            current_states, current_actions, sampled_belief
        )
        
        # Reconstruction loss
        state_loss = F.mse_loss(pred_states, next_states, reduction='mean')
        reward_loss = F.mse_loss(pred_rewards.squeeze(), next_rewards, reduction='mean')
        reconstruction_loss = state_loss + reward_loss
        
        # KL divergence loss
        belief_dist = Normal(belief_mu, torch.exp(0.5 * belief_logvar))
        prior_dist = Normal(
            self.prior_mu.expand_as(belief_mu),
            self.prior_std.expand_as(belief_mu)
        )
        kl_loss = torch.distributions.kl_divergence(belief_dist, prior_dist).sum(dim=-1).mean()
        
        # Total ELBO (negative because we minimize)
        elbo = -(reconstruction_loss + kl_loss)
        
        return {
            'elbo': elbo,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss,
            'state_loss': state_loss,
            'reward_loss': reward_loss,
            'belief_mu': belief_mu,
            'belief_logvar': belief_logvar
        }
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass for training.
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        lengths = batch['lengths']
        
        # For decoder training, we need next states/rewards
        next_states = states[:, -1, :]  # Last state in sequence
        next_rewards = rewards[:, -1]   # Last reward in sequence
        
        # Compute ELBO
        elbo_results = self.compute_elbo(
            states, actions, rewards, lengths,
            next_states, next_rewards
        )
        
        return elbo_results


def test_fixed_varibad_vae():
    """Test the fixed VariBAD VAE system with proper dimensions."""
    print("🧪 Testing Fixed VariBAD VAE Models")
    print("=" * 40)
    
    # Test with realistic S&P 500 dimensions
    state_dim = 964    # From your MDP observation space
    action_dim = 60    # 30 assets * 2 (long + short)
    latent_dim = 5
    
    batch_size = 4
    max_seq_len = 15
    
    print(f"✓ Testing with dimensions:")
    print(f"  State: {state_dim}")
    print(f"  Action: {action_dim}")
    print(f"  Latent: {latent_dim}")
    
    # Create VariBAD VAE
    varibad = VariBADVAE(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        enable_short_selling=True
    )
    
    print(f"✓ VariBAD VAE created")
    
    # Test trajectory encoding
    print("\n1. Testing trajectory encoding...")
    
    states = torch.randn(batch_size, max_seq_len, state_dim)
    actions = torch.randn(batch_size, max_seq_len, action_dim)
    rewards = torch.randn(batch_size, max_seq_len)
    lengths = torch.randint(5, max_seq_len + 1, (batch_size,))
    
    try:
        belief_mu, belief_logvar, sampled_belief = varibad.encode_trajectory(
            states, actions, rewards, lengths
        )
        
        print(f"✓ Belief parameters: μ {belief_mu.shape}, logvar {belief_logvar.shape}")
        assert belief_mu.shape == (batch_size, latent_dim)
        assert sampled_belief.shape == (batch_size, latent_dim)
        
    except Exception as e:
        print(f"❌ Trajectory encoding failed: {e}")
        return None
    
    # Test policy actions
    print("\n2. Testing portfolio policy...")
    
    current_state = torch.randn(batch_size, state_dim)
    portfolio_weights = varibad.get_policy_action(
        current_state, belief_mu, belief_logvar
    )
    
    print(f"✓ Portfolio weights: {portfolio_weights.shape}")
    assert portfolio_weights.shape == (batch_size, action_dim)
    
    # Test ELBO computation
    print("\n3. Testing ELBO computation...")
    
    batch = {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'lengths': lengths
    }
    
    try:
        results = varibad.forward(batch)
        
        print(f"✓ ELBO: {results['elbo'].item():.4f}")
        print(f"✓ Reconstruction loss: {results['reconstruction_loss'].item():.4f}")
        print(f"✓ KL loss: {results['kl_loss'].item():.4f}")
        
    except Exception as e:
        print(f"❌ ELBO computation failed: {e}")
        return None
    
    print(f"\n🎉 All tests passed! Fixed VariBAD VAE is working.")
    return varibad


if __name__ == "__main__":
    # Test the fixed models
    varibad_model = test_fixed_varibad_vae()
    
    if varibad_model:
        print(f"\n🚀 Models are ready for integration with the trainer!")
        print(f"Key fixes applied:")
        print(f"• ✅ Corrected encoder input dimension calculation")
        print(f"• ✅ Proper action dimension handling throughout")
        print(f"• ✅ Added dimension validation and debugging")
        print(f"• ✅ Fixed tensor shape mismatches")