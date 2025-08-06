# In algorithms/trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from logger_config import experiment_logger

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, env, vae, policy, config):
        self.env = env
        self.vae = vae
        self.policy = policy
        self.config = config
        
        # Training state
        self.episode_count = 0
        self.total_steps = 0
        
        # Optimizers
        self.vae_optimizer = optim.Adam(vae.parameters(), lr=config.vae_lr)
        self.policy_optimizer = optim.Adam(policy.parameters(), lr=config.policy_lr)
        
        logger.info("Trainer initialized")
        
        # Log training configuration
        if experiment_logger:
            experiment_logger.log_hyperparams({
                'trainer/vae_lr': config.vae_lr,
                'trainer/policy_lr': config.policy_lr,
                'trainer/episode_length': env.episode_length,
                'trainer/num_assets': env.num_assets,
                'trainer/max_episodes': config.max_episodes
            })
        
        self.device = config.device if hasattr(config, 'device') else 'cpu'
    
    def train_episode(self):
        """Train for one episode with logging"""
        obs = self.env.reset()
        episode_reward = 0
        episode_actions = []
        episode_observations = []
        episode_rewards = []
        
        done = False
        step = 0
        
        while not done:
            # Get latent from VAE (dummy for now)
            latent = None  # Will implement proper VAE encoding
            
            # Get action from policy
            action, value = self.policy.act(obs, latent, deterministic=False)
            
            # Take step in environment
            next_obs, reward, done, info = self.env.step(action)
            
            # Store trajectory data
            episode_actions.append(action)
            episode_observations.append(obs)
            episode_rewards.append(reward)
            
            episode_reward += reward
            step += 1
            obs = next_obs
        
        # Log episode metrics
        self._log_episode_metrics(episode_reward, step, episode_actions, episode_rewards)
        
        # Update models (placeholder)
        vae_loss = self._update_vae(episode_observations, episode_actions, episode_rewards)
        policy_loss = self._update_policy(episode_observations, episode_actions, episode_rewards)
        
        self.episode_count += 1
        self.total_steps += step
        
        return episode_reward, vae_loss, policy_loss
    
    def _log_episode_metrics(self, episode_reward, episode_length, actions, rewards):
        """Log comprehensive episode metrics"""
        if not experiment_logger:
            return
            
        # Basic episode metrics
        experiment_logger.log_scalar('training/episode_reward', episode_reward, self.episode_count)
        experiment_logger.log_scalar('training/episode_length', episode_length, self.episode_count)
        experiment_logger.log_scalar('training/total_steps', self.total_steps, self.episode_count)
        
        # Reward statistics
        rewards_tensor = torch.tensor(rewards)
        experiment_logger.log_scalar('training/reward_mean', rewards_tensor.mean().item(), self.episode_count)
        experiment_logger.log_scalar('training/reward_std', rewards_tensor.std().item(), self.episode_count)
        experiment_logger.log_histogram('training/episode_rewards', rewards_tensor, self.episode_count)
        
        # Action statistics (count decisions across episode)
        total_long = sum((action['decisions'] == 0).sum().item() for action in actions)
        total_short = sum((action['decisions'] == 1).sum().item() for action in actions)
        total_neutral = sum((action['decisions'] == 2).sum().item() for action in actions)
        
        experiment_logger.log_scalars('training/action_distribution', {
            'long_positions': total_long,
            'short_positions': total_short, 
            'neutral_positions': total_neutral
        }, self.episode_count)
        
        logger.info(f"Episode {self.episode_count}: reward={episode_reward:.4f}, "
                   f"steps={episode_length}, long={total_long}, short={total_short}")
    
    # In algorithms/trainer.py - complete _update_vae
    def _update_vae(self, observations, actions, rewards):
        """Update VAE with logging"""
        # Convert trajectory to tensors
        obs_seq = torch.stack([torch.tensor(obs, dtype=torch.float32) for obs in observations]).unsqueeze(0).to(self.device)
        reward_seq = torch.tensor(rewards, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
        
        # Convert hierarchical actions to flat vectors
        action_seq = []
        for action in actions:
            # Only flatten the actual action tensors, maintain consistent size
            decisions_flat = action['decisions'].float()  # (3,) -> (3,)
            long_flat = action['long_weights'].float()    # (3,) -> (3,) 
            short_flat = action['short_weights'].float()  # (3,) -> (3,)
            
            # Create fixed-size representation: pad to num_assets
            action_flat = torch.zeros(self.config.num_assets * 3)
            action_flat[:self.config.num_assets] = decisions_flat
            action_flat[self.config.num_assets:2*self.config.num_assets] = long_flat
            action_flat[2*self.config.num_assets:] = short_flat
            
            action_seq.append(action_flat)
        action_seq = torch.stack(action_seq).unsqueeze(0)  # (1, seq_len, 90)
        
        reward_seq = torch.tensor(rewards, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        
        # Forward pass and compute loss
        total_loss, loss_components = self.vae.compute_loss(obs_seq, action_seq, reward_seq)
        
        # Backward pass
        self.vae_optimizer.zero_grad()
        total_loss.backward()
        self.vae_optimizer.step()
        
        logger.debug(f"VAE updated: total_loss={total_loss.item():.4f}")
        
        return total_loss.item()
    
    # In algorithms/trainer.py - complete _update_policy
    def _update_policy(self, observations, actions, rewards):
        """Update policy with logging"""
        if len(observations) < 2:  # Need at least 2 steps for advantage estimation
            return 0.0
            
        # Convert to tensors
        obs_tensor = torch.stack([torch.tensor(obs, dtype=torch.float32) for obs in observations[:-1]])  # All but last
        next_obs_tensor = torch.stack([torch.tensor(obs, dtype=torch.float32) for obs in observations[1:]])  # All but first
        rewards_tensor = torch.tensor(rewards[:-1], dtype=torch.float32)  # Match length
        
        # Get latent representations (placeholder - will use VAE later)
        latent = None  # For now, policy works without latent
        
        # Forward pass through policy for all timesteps
        batch_size = obs_tensor.shape[0]
        values = []
        action_log_probs = []
        
        for t in range(batch_size):
            # Get value and action log prob for this timestep
            with torch.no_grad():
                _, value = self.policy.act(obs_tensor[t:t+1], latent, deterministic=True)
                values.append(value)
            
            # Evaluate action that was actually taken
            taken_action = {
                'decisions': actions[t]['decisions'].unsqueeze(0),
                'long_weights': actions[t]['long_weights'].unsqueeze(0),
                'short_weights': actions[t]['short_weights'].unsqueeze(0)
            }
            
            _, log_prob, _ = self.policy.evaluate_actions(obs_tensor[t:t+1], latent, taken_action)
            action_log_probs.append(log_prob)
        
        values = torch.cat(values)
        action_log_probs = torch.cat(action_log_probs)
        
        # Compute advantages using simple TD error
        advantages = torch.zeros_like(rewards_tensor)
        for t in range(len(rewards_tensor)):
            if t == len(rewards_tensor) - 1:
                # Last step
                advantages[t] = rewards_tensor[t] - values[t]
            else:
                # TD error: r + γV(s') - V(s)
                advantages[t] = rewards_tensor[t] + 0.99 * values[t+1] - values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss (REINFORCE with baseline)
        policy_loss = -(action_log_probs * advantages.detach()).mean()
        
        # Value loss
        returns = torch.zeros_like(rewards_tensor)
        running_return = 0
        for t in reversed(range(len(rewards_tensor))):
            running_return = rewards_tensor[t] + 0.99 * running_return
            returns[t] = running_return
        
        value_loss = F.mse_loss(values, returns.detach())
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Backward pass
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        self.policy_optimizer.step()
        
        # Log policy metrics
        if experiment_logger:
            experiment_logger.log_scalars('policy/loss_components', {
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'total': total_loss.item()
            }, self.episode_count)
            
            experiment_logger.log_scalar('policy/advantage_mean', advantages.mean().item(), self.episode_count)
            experiment_logger.log_scalar('policy/advantage_std', advantages.std().item(), self.episode_count)
        
        logger.debug(f"Policy updated: total_loss={total_loss.item():.4f}")
        
        return total_loss.item()