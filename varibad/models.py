"""
VariBAD models and portfolio environment
Consolidated from varibad/core/models.py and varibad/core/environment.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Dict, Tuple, Optional, List
from torch.distributions import Normal


class TrajectoryEncoder(nn.Module):
    """Encoder q_φ(m|τ:t): Trajectory sequences → Belief parameters"""
    
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int = 5, 
                hidden_dim: int = 128, num_layers: int = 2, rnn_type: str = 'GRU', 
                dropout: float = 0.1, bidirectional: bool = False):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # Input: [state, action, reward]
        self.input_dim = state_dim + action_dim + 1
        
        # RNN for trajectory processing
        if rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        else:  # Default to GRU
            self.rnn = nn.GRU(
                input_size=self.input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        
        # Adjust output dimension for bidirectional
        rnn_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Output layers
        self.fc_mu = nn.Linear(rnn_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(rnn_output_dim, latent_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.xavier_uniform_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)
    
    def forward(self, states, actions, rewards, lengths):
        """Forward pass: trajectory → belief parameters"""
        
        # Concatenate inputs
        rewards_expanded = rewards.unsqueeze(-1)
        trajectory_input = torch.cat([states, actions, rewards_expanded], dim=-1)
        
        # Check for NaN inputs
        if torch.isnan(trajectory_input).any():
            # Return zero belief
            batch_size = states.shape[0]
            mu = torch.zeros(batch_size, self.latent_dim, device=states.device)
            logvar = torch.zeros(batch_size, self.latent_dim, device=states.device)
            return mu, logvar
        
        # Pack sequences
        packed_input = nn.utils.rnn.pack_padded_sequence(
            trajectory_input, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Process through RNN
        packed_output, hidden = self.rnn(packed_input)
        
        # Use final hidden state
        final_hidden = hidden[-1]  # Last layer
        
        # Compute belief parameters with stability
        mu = self.fc_mu(final_hidden)
        logvar = self.fc_logvar(final_hidden)
        
        # Clamp for numerical stability
        mu = torch.clamp(mu, -10, 10)
        logvar = torch.clamp(logvar, -10, 10)
        
        return mu, logvar


class TrajectoryDecoder(nn.Module):
    """Decoder p_θ(τ_{t+1:H+}|m): Belief → Future predictions"""
    
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int = 5, hidden_dim: int = 128):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # Input: [state, action, belief]
        input_dim = state_dim + action_dim + latent_dim
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Prediction heads
        self.state_predictor = nn.Linear(hidden_dim, state_dim)
        self.reward_predictor = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, current_states, current_actions, task_embeddings):
        """Forward pass: current state/action + belief → predictions"""
        
        # Concatenate inputs
        decoder_input = torch.cat([current_states, current_actions, task_embeddings], dim=-1)
        
        # Process through shared layers
        features = self.shared_layers(decoder_input)
        
        # Generate predictions
        predicted_states = self.state_predictor(features)
        predicted_rewards = self.reward_predictor(features)
        
        return predicted_states, predicted_rewards


class VariBADPolicy(nn.Module):
    """Policy π_ψ(a_t|s_t, q(m|τ:t)): State + Belief → Portfolio weights"""
    
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int = 5, 
                 hidden_dim: int = 256, enable_short_selling: bool = True):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.enable_short_selling = enable_short_selling
        
        if enable_short_selling:
            self.num_assets = action_dim // 2
        else:
            self.num_assets = action_dim
        
        # Input: [state, belief_mu, belief_std]
        input_dim = state_dim + 2 * latent_dim
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output heads
        self.long_head = nn.Linear(hidden_dim, self.num_assets)
        if enable_short_selling:
            self.short_head = nn.Linear(hidden_dim, self.num_assets)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, states, belief_mu, belief_logvar):
        """Forward pass: state + belief → portfolio weights"""
        
        # Combine inputs
        belief_std = torch.exp(0.5 * belief_logvar)
        policy_input = torch.cat([states, belief_mu, belief_std], dim=-1)
        
        # Process through shared layers
        features = self.shared_layers(policy_input)
        
        # Generate portfolio weights
        long_logits = self.long_head(features)
        
        if self.enable_short_selling:
            short_logits = self.short_head(features)
            
            # Apply constraints
            long_weights = F.softmax(long_logits, dim=-1)  # Sum to 1
            short_weights = -F.sigmoid(short_logits) * 0.3  # Max 30% short
            
            portfolio_weights = torch.cat([long_weights, short_weights], dim=-1)
        else:
            portfolio_weights = F.softmax(long_logits, dim=-1)
        
        return portfolio_weights


class VariBADVAE(nn.Module):
    """Complete VariBAD VAE system"""
    
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int = 5, 
                encoder_hidden: int = 128, decoder_hidden: int = 128, 
                policy_hidden: int = 256, enable_short_selling: bool = True,
                encoder_layers: int = 2, encoder_rnn_type: str = 'GRU',
                encoder_dropout: float = 0.1, encoder_bidirectional: bool = False):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.enable_short_selling = enable_short_selling
        
        # Components with enhanced parameters
        self.encoder = TrajectoryEncoder(
            state_dim, action_dim, latent_dim, encoder_hidden,
            encoder_layers, encoder_rnn_type, encoder_dropout, encoder_bidirectional
        )
        self.decoder = TrajectoryDecoder(state_dim, action_dim, latent_dim, decoder_hidden)
        self.policy = VariBADPolicy(state_dim, action_dim, latent_dim, policy_hidden, enable_short_selling)
        
        # Prior distribution
        self.register_buffer('prior_mu', torch.zeros(latent_dim))
        self.register_buffer('prior_std', torch.ones(latent_dim))
    
    def encode_trajectory(self, states, actions, rewards, lengths):
        """Encode trajectory to belief"""
        belief_mu, belief_logvar = self.encoder(states, actions, rewards, lengths)
        # Sample belief
        std = torch.exp(0.5 * belief_logvar)
        eps = torch.randn_like(std)
        sampled_belief = belief_mu + eps * std
        return belief_mu, belief_logvar, sampled_belief
    
    def get_policy_action(self, state, belief_mu, belief_logvar):
        """Get portfolio action from policy"""
        return self.policy(state, belief_mu, belief_logvar)
    
    def compute_elbo(self, states, actions, rewards, lengths, next_states, next_rewards):
        """Compute ELBO loss with numerical stability"""
        batch_size = states.shape[0]
        
        # Encode trajectory
        belief_mu, belief_logvar, sampled_belief = self.encode_trajectory(states, actions, rewards, lengths)
        
        # Clamp belief parameters for numerical stability
        belief_mu = torch.clamp(belief_mu, -10, 10)
        belief_logvar = torch.clamp(belief_logvar, -10, 10)
        
        # Check for NaN values
        if torch.isnan(belief_mu).any() or torch.isnan(belief_logvar).any():
            # Return dummy loss to continue training
            return {
                'elbo': torch.tensor(0.0, device=belief_mu.device),
                'reconstruction_loss': torch.tensor(0.0, device=belief_mu.device),
                'kl_loss': torch.tensor(0.0, device=belief_mu.device),
                'state_loss': torch.tensor(0.0, device=belief_mu.device),
                'reward_loss': torch.tensor(0.0, device=belief_mu.device)
            }
        
        # Use last state/action for decoding
        last_indices = lengths - 1
        current_states = states[torch.arange(batch_size), last_indices]
        current_actions = actions[torch.arange(batch_size), last_indices]
        
        # Decode predictions
        pred_states, pred_rewards = self.decoder(current_states, current_actions, sampled_belief)
        
        # Reconstruction loss with clamping
        state_loss = F.mse_loss(pred_states, next_states, reduction='mean')
        reward_loss = F.mse_loss(pred_rewards.squeeze(-1), next_rewards, reduction='mean')
        reconstruction_loss = state_loss + reward_loss
        
        # KL divergence with numerical stability
        try:
            belief_dist = Normal(belief_mu, torch.exp(0.5 * belief_logvar) + 1e-6)
            prior_dist = Normal(self.prior_mu.expand_as(belief_mu), self.prior_std.expand_as(belief_mu))
            kl_loss = torch.distributions.kl_divergence(belief_dist, prior_dist).sum(dim=-1).mean()
            
            # Check for NaN in KL
            if torch.isnan(kl_loss):
                kl_loss = torch.tensor(0.0, device=belief_mu.device)
                
        except Exception:
            kl_loss = torch.tensor(0.0, device=belief_mu.device)
        
        # ELBO (negative because we minimize)
        elbo = -(reconstruction_loss + kl_loss)
        
        return {
            'elbo': elbo,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss,
            'state_loss': state_loss,
            'reward_loss': reward_loss
        }


class PortfolioEnvironment(gym.Env):
    """Portfolio optimization environment"""
    
    def __init__(self, data: pd.DataFrame, episode_length: int = 30, 
                 enable_short_selling: bool = True, max_short_ratio: float = 0.3,
                 transaction_cost: float = 0.001, lookback_window: int = 20):
        super().__init__()
        
        self.data = data.sort_values(['date', 'ticker']).reset_index(drop=True)
        self.episode_length = episode_length
        self.enable_short_selling = enable_short_selling
        self.max_short_ratio = max_short_ratio
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        
        # Extract dates and tickers
        self.dates = sorted(self.data['date'].unique())
        self.tickers = sorted(self.data['ticker'].unique())
        self.n_assets = len(self.tickers)
        
        # Identify feature columns
        self.asset_features = [col for col in self.data.columns if col.endswith('_norm')]
        self.market_features = ['market_return', 'excess_returns', 'volatility_5d', 'volatility_20d']
        self.market_features = [col for col in self.market_features if col in self.data.columns]
        
        # Define spaces
        self._define_spaces()
        
        # Initialize episode state
        self.reset()
    
    def _define_spaces(self):
        """Define action and observation spaces"""
        
        # Action space
        if self.enable_short_selling:
            action_dim = 2 * self.n_assets  # long + short weights
            action_low = np.concatenate([np.zeros(self.n_assets), 
                                       np.full(self.n_assets, -self.max_short_ratio)])
            action_high = np.concatenate([np.ones(self.n_assets), np.zeros(self.n_assets)])
        else:
            action_dim = self.n_assets
            action_low = np.zeros(self.n_assets)
            action_high = np.ones(self.n_assets)
        
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        
        # Observation space
        asset_feature_dim = len(self.asset_features) * self.n_assets
        market_feature_dim = len(self.market_features)
        account_feature_dim = action_dim
        
        obs_dim = asset_feature_dim + market_feature_dim + account_feature_dim
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)
    
    def _get_date_data(self, date_idx: int) -> pd.DataFrame:
        """Get data for specific date"""
        date = self.dates[date_idx]
        return self.data[self.data['date'] == date].set_index('ticker').reindex(self.tickers)
    
    def _construct_state(self, date_idx: int, previous_weights: np.ndarray) -> np.ndarray:
        """Construct observation state"""
        
        current_data = self._get_date_data(date_idx)
        
        # Asset features
        asset_features = []
        for ticker in self.tickers:
            if ticker in current_data.index:
                features = current_data.loc[ticker, self.asset_features].values
                features = np.nan_to_num(features, nan=0.0)
            else:
                features = np.zeros(len(self.asset_features))
            asset_features.append(features)
        asset_features = np.concatenate(asset_features)
        
        # Market features
        if len(current_data) > 0:
            market_features = current_data.iloc[0][self.market_features].values
            market_features = np.nan_to_num(market_features, nan=0.0)
        else:
            market_features = np.zeros(len(self.market_features))
        
        # Account features (previous weights)
        account_features = previous_weights.copy()
        
        # Combine
        state = np.concatenate([asset_features, market_features, account_features])
        return state.astype(np.float32)
    
    def _validate_action(self, action: np.ndarray) -> np.ndarray:
        """Validate and normalize portfolio action"""
        action = np.array(action, dtype=np.float32)
        
        if self.enable_short_selling:
            # Split long and short
            long_weights = action[:self.n_assets]
            short_weights = action[self.n_assets:]
            
            # Normalize long weights to sum to 1
            long_weights = np.maximum(long_weights, 0.0)
            if long_weights.sum() > 0:
                long_weights = long_weights / long_weights.sum()
            else:
                long_weights = np.ones(self.n_assets) / self.n_assets
            
            # Constrain short weights
            short_weights = np.minimum(short_weights, 0.0)
            if short_weights.sum() < -self.max_short_ratio:
                short_weights = short_weights * (self.max_short_ratio / abs(short_weights.sum()))
            
            return np.concatenate([long_weights, short_weights])
        else:
            # Long-only
            action = np.maximum(action, 0.0)
            if action.sum() > 0:
                action = action / action.sum()
            else:
                action = np.ones(self.n_assets) / self.n_assets
            return action
    
    def _execute_action(self, action: np.ndarray, date_idx: int) -> Tuple[float, float]:
        """Execute portfolio action and return performance"""
        
        current_data = self._get_date_data(date_idx)
        
        if self.enable_short_selling:
            long_weights = action[:self.n_assets]
            short_weights = action[self.n_assets:]
            
            portfolio_return = 0.0
            for i, ticker in enumerate(self.tickers):
                if ticker in current_data.index:
                    asset_return = current_data.loc[ticker, 'returns']
                    if not np.isnan(asset_return):
                        portfolio_return += long_weights[i] * asset_return
                        portfolio_return += short_weights[i] * asset_return
            
            current_net_weights = long_weights + short_weights
        else:
            portfolio_return = 0.0
            for i, ticker in enumerate(self.tickers):
                if ticker in current_data.index:
                    asset_return = current_data.loc[ticker, 'returns']
                    if not np.isnan(asset_return):
                        portfolio_return += action[i] * asset_return
            
            current_net_weights = action.copy()
        
        # Transaction costs
        if hasattr(self, 'previous_net_weights'):
            weight_changes = np.abs(current_net_weights - self.previous_net_weights)
            transaction_cost = np.sum(weight_changes) * self.transaction_cost
        else:
            transaction_cost = 0.0
        
        self.previous_net_weights = current_net_weights.copy()
        return portfolio_return - transaction_cost, transaction_cost
    
    def _calculate_dsr_reward(self, return_rate: float) -> float:
        """Calculate Differential Sharpe Ratio reward"""
        eta = 0.01
        excess_return = return_rate - 0.02/252  # Daily risk-free rate
        
        # Update exponential moving averages
        delta_alpha = excess_return - self.alpha
        self.alpha = self.alpha + eta * delta_alpha
        
        delta_beta = excess_return**2 - self.beta
        self.beta = self.beta + eta * delta_beta
        
        # Calculate DSR
        variance_estimate = self.beta - self.alpha**2
        if variance_estimate > 1e-6:
            denominator = variance_estimate**(3/2)
            numerator = self.beta * delta_alpha - 0.5 * self.alpha * delta_beta
            dsr = numerator / denominator
        else:
            dsr = excess_return * 10
        
        return np.clip(dsr, -10.0, 10.0)
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment"""
        if seed is not None:
            np.random.seed(seed)
        
        # Sample episode start
        min_start = self.lookback_window
        max_start = len(self.dates) - self.episode_length - 1
        self.episode_start_idx = np.random.randint(min_start, max_start)
        self.current_step = 0
        
        # Initialize portfolio
        if self.enable_short_selling:
            initial_weights = np.concatenate([
                np.ones(self.n_assets) / self.n_assets,
                np.zeros(self.n_assets)
            ])
        else:
            initial_weights = np.ones(self.n_assets) / self.n_assets
        
        self.current_weights = initial_weights.copy()
        
        # Initialize DSR tracking
        self.alpha = 0.0
        self.beta = 0.01
        
        # Construct initial state
        current_date_idx = self.episode_start_idx + self.current_step
        state = self._construct_state(current_date_idx, self.current_weights)
        
        return state
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one environment step"""
        
        # Validate action
        action = self._validate_action(np.array(action))
        
        # Execute action
        current_date_idx = self.episode_start_idx + self.current_step
        portfolio_return, transaction_cost = self._execute_action(action, current_date_idx + 1)
        
        # Calculate reward
        reward = self._calculate_dsr_reward(portfolio_return)
        
        # Update step
        self.current_step += 1
        done = self.current_step >= self.episode_length
        
        # Next state
        if not done:
            next_date_idx = self.episode_start_idx + self.current_step
            next_state = self._construct_state(next_date_idx, self.current_weights)
        else:
            next_state = np.zeros_like(self.observation_space.sample())
        
        # Update weights
        self.current_weights = action.copy()
        
        # Info
        info = {
            'portfolio_return': portfolio_return,
            'transaction_cost': transaction_cost,
            'dsr_reward': reward,
            'step': self.current_step,
            'date': self.dates[current_date_idx + 1].strftime('%Y-%m-%d') if current_date_idx + 1 < len(self.dates) else 'end'
        }
        
        return next_state, reward, done, info