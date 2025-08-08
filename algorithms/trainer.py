# File: algorithms/trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import logging
from collections import deque

logger = logging.getLogger(__name__)

class PPOTrainer:
    """
    PPO Trainer for VariBAD Portfolio Optimization.
    Handles both policy updates and VAE training.
    """
    
    def __init__(self, env, policy, vae, config, logger=None):
        self.env = env
        self.policy = policy
        self.vae = vae
        self.config = config
        self.exp_logger = logger
        
        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.device = torch.device(config.device)
        
        # Optimizers
        self.policy_optimizer = Adam(policy.parameters(), lr=config.policy_lr)
        self.vae_optimizer = Adam(vae.parameters(), lr=config.vae_lr)
        
        # Experience buffer for PPO
        self.experience_buffer = ExperienceBuffer(config.batch_size)
        
        # VAE training buffer (separate from PPO buffer)
        self.vae_buffer = deque(maxlen=1000)  # Store recent trajectories
        
        # Training statistics
        self.policy_losses = deque(maxlen=100)
        self.vae_losses = deque(maxlen=100)
        self.episode_rewards = deque(maxlen=100)
        
        logger.info("PPO Trainer initialized")
        logger.info(f"Policy LR: {config.policy_lr}, VAE LR: {config.vae_lr}")
        logger.info(f"PPO epochs: {config.ppo_epochs}, clip ratio: {config.ppo_clip_ratio}")
    
    def train_episode(self):
        """Train for one episode and return results"""
        # Collect episode trajectory
        trajectory = self.collect_trajectory()
        
        # Add to VAE training buffer
        self.vae_buffer.append(trajectory)
        
        # Add to PPO experience buffer
        self.experience_buffer.add_trajectory(trajectory)
        
        # Update models if we have enough data
        policy_loss = 0
        vae_loss = 0
        
        # PPO update
        if self.experience_buffer.is_ready():
            policy_loss = self.update_policy()
            self.experience_buffer.clear()
        
        # VAE update (less frequent)
        if self.episode_count % self.config.vae_update_freq == 0 and len(self.vae_buffer) >= self.config.vae_batch_size:
            vae_loss = self.update_vae()
        
        # Update statistics
        episode_reward = sum(trajectory['rewards'])
        self.episode_rewards.append(episode_reward)
        if policy_loss > 0:
            self.policy_losses.append(policy_loss)
        if vae_loss > 0:
            self.vae_losses.append(vae_loss)
        
        self.episode_count += 1
        
        # Logging
        if self.exp_logger and self.episode_count % self.config.log_interval == 0:
            self._log_training_stats()
        
        return {
            'episode_reward': episode_reward,
            'policy_loss': policy_loss,
            'vae_loss': vae_loss,
            'episode_length': len(trajectory['rewards'])
        }
    
    def collect_trajectory(self):
        """Collect a single episode trajectory"""
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'latents': [],
            'dones': []
        }
        
        # Reset environment
        obs = self.env.reset()  # (N, F)
        obs_tensor = torch.ascontiguous_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, N, F)
        
        # Initialize latent (will be updated by VAE encoder)
        latent = torch.zeros(1, self.config.latent_dim, device=self.device)
        
        # Episode loop
        done = False
        step = 0
        
        while not done and step < self.config.max_horizon:
            with torch.no_grad():
                # Get action and value from policy
                action, value = self.policy.act(obs_tensor, latent, deterministic=False)
                
                # Evaluate action to get log probability
                _, log_prob, _ = self.policy.evaluate_actions(obs_tensor, latent, action)
                
            # Take environment step
            action_cpu = action.squeeze(0).cpu().numpy()  # (N,)
            next_obs, reward, done, info = self.env.step(action_cpu)
            
            # Store transition
            trajectory['observations'].append(obs_tensor.squeeze(0))  # (N, F)
            trajectory['actions'].append(action.squeeze(0))           # (N,)
            trajectory['rewards'].append(reward)
            trajectory['values'].append(value.squeeze())             # scalar
            trajectory['log_probs'].append(log_prob.squeeze())       # scalar
            trajectory['latents'].append(latent.squeeze(0))           # (latent_dim,)
            trajectory['dones'].append(done)
            
            # Update for next step
            if not done:
                obs_tensor = torch.ascontiguous_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                
                # Update latent using VAE encoder with trajectory context
                # For now, keep latent constant during episode
                # TODO: Implement proper trajectory encoding for latent updates
                
            step += 1
            self.total_steps += 1
        
        # Convert lists to tensors
        trajectory['observations'] = torch.stack(trajectory['observations'])  # (T, N, F)
        trajectory['actions'] = torch.stack(trajectory['actions'])            # (T, N)
        trajectory['rewards'] = torch.tensor(trajectory['rewards'], device=self.device)  # (T,)
        trajectory['values'] = torch.stack(trajectory['values'])              # (T,)
        trajectory['log_probs'] = torch.stack(trajectory['log_probs'])        # (T,)
        trajectory['latents'] = torch.stack(trajectory['latents'])            # (T, latent_dim)
        
        return trajectory
    
    def compute_advantages(self, trajectory):
        """Compute GAE advantages"""
        rewards = trajectory['rewards']         # (T,)
        values = trajectory['values']           # (T,)
        dones = trajectory['dones']
        
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # GAE computation
        gae = 0
        next_value = 0  # Value of terminal state is 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.config.discount_factor * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.discount_factor * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_value = values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update_policy(self):
        """Update policy using PPO"""
        all_trajectories = self.experience_buffer.get_all()
        total_loss = 0
        
        # Compute advantages for all trajectories
        batch_obs = []
        batch_actions = []
        batch_latents = []
        batch_advantages = []
        batch_returns = []
        batch_old_log_probs = []
        
        for trajectory in all_trajectories:
            advantages, returns = self.compute_advantages(trajectory)
            
            batch_obs.append(trajectory['observations'])
            batch_actions.append(trajectory['actions'])
            batch_latents.append(trajectory['latents'])
            batch_advantages.append(advantages)
            batch_returns.append(returns)
            batch_old_log_probs.append(trajectory['log_probs'])
        
        # Concatenate all data
        batch_obs = torch.cat(batch_obs, dim=0)                    # (batch_size, N, F)
        batch_actions = torch.cat(batch_actions, dim=0)            # (batch_size, N)
        batch_latents = torch.cat(batch_latents, dim=0)            # (batch_size, latent_dim)
        batch_advantages = torch.cat(batch_advantages, dim=0)      # (batch_size,)
        batch_returns = torch.cat(batch_returns, dim=0)           # (batch_size,)
        batch_old_log_probs = torch.cat(batch_old_log_probs, dim=0)  # (batch_size,)
        
        # PPO epochs
        for epoch in range(self.config.ppo_epochs):
            # Forward pass
            values, log_probs, entropy = self.policy.evaluate_actions(
                batch_obs, batch_latents, batch_actions
            )
            
            values = values.squeeze(-1)    # (batch_size,)
            log_probs = log_probs.squeeze(-1)  # (batch_size,)
            entropy = entropy.mean()       # scalar
            
            # Compute PPO loss
            ratio = torch.exp(log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip_ratio, 
                               1 + self.config.ppo_clip_ratio) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, batch_returns)
            
            # Total loss
            loss = (policy_loss + 
                   self.config.value_loss_coef * value_loss - 
                   self.config.entropy_coef * entropy)
            
            # Update
            self.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.policy_optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / self.config.ppo_epochs
    
    def update_vae(self):
        """Update VAE using recent trajectories"""
        if len(self.vae_buffer) < self.config.vae_batch_size:
            return 0
        
        # Sample batch of trajectories
        indices = np.random.choice(len(self.vae_buffer), self.config.vae_batch_size, replace=False)
        batch_trajectories = [self.vae_buffer[i] for i in indices]
        
        # Prepare sequences for VAE
        obs_sequences = []
        action_sequences = []
        reward_sequences = []
        
        for trajectory in batch_trajectories:
            # Pad sequences to same length if needed
            seq_len = len(trajectory['rewards'])
            if seq_len >= 10:  # Minimum sequence length for VAE training
                obs_sequences.append(trajectory['observations'].unsqueeze(0))    # (1, T, N, F)
                action_sequences.append(trajectory['actions'].unsqueeze(0))      # (1, T, N)
                reward_sequences.append(trajectory['rewards'].unsqueeze(0).unsqueeze(-1))  # (1, T, 1)
        
        if not obs_sequences:
            return 0
        
        # Batch sequences (pad if necessary)
        max_len = max(seq.shape[1] for seq in obs_sequences)
        
        batch_obs = torch.zeros(len(obs_sequences), max_len, *obs_sequences[0].shape[2:], device=self.device)
        batch_actions = torch.zeros(len(action_sequences), max_len, action_sequences[0].shape[2], device=self.device)
        batch_rewards = torch.zeros(len(reward_sequences), max_len, 1, device=self.device)
        
        for i, (obs_seq, act_seq, rew_seq) in enumerate(zip(obs_sequences, action_sequences, reward_sequences)):
            seq_len = obs_seq.shape[1]
            batch_obs[i, :seq_len] = obs_seq[0]
            batch_actions[i, :seq_len] = act_seq[0]
            batch_rewards[i, :seq_len] = rew_seq[0]
        
        # VAE forward pass and loss
        vae_loss, loss_components = self.vae.compute_loss(
            batch_obs, batch_actions, batch_rewards, beta=self.config.vae_beta
        )
        
        # Update VAE
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.config.max_grad_norm)
        self.vae_optimizer.step()
        
        return vae_loss.item()
    
    def _log_training_stats(self):
        """Log training statistics"""
        if not self.exp_logger:
            return
        
        # Episode statistics
        if self.episode_rewards:
            self.exp_logger.log_scalar('train/episode_reward_mean', 
                                     np.mean(self.episode_rewards), self.episode_count)
            self.exp_logger.log_scalar('train/episode_reward_std', 
                                     np.std(self.episode_rewards), self.episode_count)
        
        # Loss statistics
        if self.policy_losses:
            self.exp_logger.log_scalar('train/policy_loss', 
                                     np.mean(self.policy_losses), self.episode_count)
        
        if self.vae_losses:
            self.exp_logger.log_scalar('train/vae_loss', 
                                     np.mean(self.vae_losses), self.episode_count)
        
        # General stats
        self.exp_logger.log_scalar('train/episode_count', self.episode_count, self.episode_count)
        self.exp_logger.log_scalar('train/total_steps', self.total_steps, self.episode_count)
    
    def get_state(self):
        """Get trainer state for checkpointing"""
        return {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'vae_optimizer': self.vae_optimizer.state_dict()
        }
    
    def load_state(self, state):
        """Load trainer state from checkpoint"""
        self.episode_count = state.get('episode_count', 0)
        self.total_steps = state.get('total_steps', 0)
        if 'policy_optimizer' in state:
            self.policy_optimizer.load_state_dict(state['policy_optimizer'])
        if 'vae_optimizer' in state:
            self.vae_optimizer.load_state_dict(state['vae_optimizer'])


class ExperienceBuffer:
    """Buffer for storing PPO training data"""
    
    def __init__(self, min_batch_size=64):
        self.min_batch_size = min_batch_size
        self.trajectories = []
        self.total_steps = 0
    
    def add_trajectory(self, trajectory):
        """Add a complete trajectory"""
        self.trajectories.append(trajectory)
        self.total_steps += len(trajectory['rewards'])
    
    def is_ready(self):
        """Check if buffer has enough data for training"""
        return self.total_steps >= self.min_batch_size
    
    def get_all(self):
        """Get all trajectories"""
        return self.trajectories
    
    def clear(self):
        """Clear buffer"""
        self.trajectories = []
        self.total_steps = 0