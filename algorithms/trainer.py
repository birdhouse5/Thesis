import torch
import logging
from typing import Dict, List, Tuple, Any
import numpy as np
from logger_config import setup_experiment_logging

class Trainer:
    def __init__(self, env, vae, policy, config, exp_name: str):
        # Core components
        self.env = env
        self.vae = vae  # Contains encoder + decoder
        self.policy = policy
        self.config = config
        
        # Optimizers
        self.vae_optimizer = torch.optim.Adam(
            self.vae.parameters(), 
            lr=config.vae_lr
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=config.policy_lr
        )
        
        # Logging
        self.logger = setup_experiment_logging(exp_name)
        self.logger.log_hyperparams(config.__dict__)
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
        self.current_trajectory = None
        self.current_mdp = None
        
        logging.info("VariBAD Trainer initialized")
    
    def sample_mdp(self):
        """Sample a new task from p(M)"""
        # This depends on your environment implementation
        # Could be: self.current_mdp = self.env.sample_task()
        # Or: self.env.reset_task()  # if env handles task sampling internally
        self.current_mdp = self.env.sample_task()
        logging.debug(f"Sampled new MDP: {self.current_mdp}")
    
    def collect_trajectory(self) -> Dict[str, List]:
        """Rollout trajectory of length H+ using current policy + encoder"""
        trajectory = {
            'states': [],
            'actions': [], 
            'rewards': [],
            'dones': []
        }
        
        # Reset environment for new trajectory
        state = self.env.reset()
        trajectory['states'].append(state)
        
        # Initialize encoder hidden state
        encoder_hidden = self.vae.encoder.init_hidden()
        
        total_reward = 0
        
        for t in range(self.config.horizon_plus):
            # Encode trajectory up to current timestep
            traj_slice = self._get_trajectory_slice(trajectory, t)
            posterior, encoder_hidden = self.vae.encoder(traj_slice, encoder_hidden)
            
            # Sample latent and get action
            latent = self._sample_latent(posterior)
            action = self.policy(state, latent)
            
            # Environment step
            next_state, reward, done, info = self.env.step(action)
            
            # Store transition
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['dones'].append(done)
            trajectory['states'].append(next_state)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Compute returns
        trajectory['returns'] = self._compute_returns(trajectory['rewards'])
        
        # Log episode metrics
        self.logger.log_scalar('train/episode_return', total_reward, self.episode_count)
        self.logger.log_scalar('train/episode_length', len(trajectory['actions']), self.episode_count)
        
        self.current_trajectory = trajectory
        self.episode_count += 1
        
        logging.info(f"Episode {self.episode_count}: Return={total_reward:.2f}, Length={len(trajectory['actions'])}")
        
        return trajectory
    
    def compute_vae_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute VAE loss (ELBO) from subsampled timesteps"""
        if self.current_trajectory is None:
            raise ValueError("No trajectory collected")
        
        # Subsample timesteps for efficiency
        max_t = len(self.current_trajectory['actions'])
        sampled_timesteps = np.random.choice(
            max_t, 
            size=min(self.config.elbo_subsample_k, max_t), 
            replace=False
        )
        
        total_reconstruction_loss = 0
        total_kl_loss = 0
        
        for t in sampled_timesteps:
            # Encode trajectory up to timestep t
            traj_slice = self._get_trajectory_slice(self.current_trajectory, t)
            posterior_t = self.vae.encoder(traj_slice)
            
            # Sample latent
            latent = self._sample_latent(posterior_t)
            
            # Decode remaining trajectory
            remaining_actions = self.current_trajectory['actions'][t:]
            pred_states, pred_rewards = self.vae.decoder(latent, remaining_actions)
            
            # Actual future states/rewards
            actual_states = self.current_trajectory['states'][t+1:t+1+len(remaining_actions)]
            actual_rewards = self.current_trajectory['rewards'][t:t+len(remaining_actions)]
            
            # Reconstruction loss
            state_loss = torch.nn.functional.mse_loss(pred_states, torch.tensor(actual_states))
            reward_loss = torch.nn.functional.mse_loss(pred_rewards, torch.tensor(actual_rewards))
            reconstruction_loss = state_loss + reward_loss
            
            # KL loss
            prior = self._get_prior(t)
            kl_loss = self._compute_kl_divergence(posterior_t, prior)
            
            total_reconstruction_loss += reconstruction_loss
            total_kl_loss += kl_loss
        
        # Average over sampled timesteps
        avg_reconstruction_loss = total_reconstruction_loss / len(sampled_timesteps)
        avg_kl_loss = total_kl_loss / len(sampled_timesteps)
        
        # Total VAE loss (ELBO = reconstruction - KL)
        vae_loss = avg_reconstruction_loss + self.config.kl_weight * avg_kl_loss
        
        # Log VAE metrics
        metrics = {
            'reconstruction_loss': avg_reconstruction_loss.item(),
            'kl_loss': avg_kl_loss.item(),
            'vae_loss': vae_loss.item()
        }
        
        return vae_loss, metrics
    
    def compute_policy_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute policy loss using standard RL objective"""
        if self.current_trajectory is None:
            raise ValueError("No trajectory collected")
        
        policy_loss = 0
        num_steps = len(self.current_trajectory['actions'])
        
        for t in range(num_steps):
            # Re-encode trajectory up to timestep t
            traj_slice = self._get_trajectory_slice(self.current_trajectory, t)
            posterior_t = self.vae.encoder(traj_slice)
            latent = self._sample_latent(posterior_t)
            
            # Get action probability
            state = self.current_trajectory['states'][t]
            action = self.current_trajectory['actions'][t]
            log_prob = self.policy.log_prob(state, latent, action)
            
            # Advantage (using return as advantage for simplicity)
            advantage = self.current_trajectory['returns'][t]
            
            # Policy gradient loss
            policy_loss += -log_prob * advantage
        
        avg_policy_loss = policy_loss / num_steps
        
        metrics = {
            'policy_loss': avg_policy_loss.item(),
            'avg_return': np.mean(self.current_trajectory['returns'])
        }
        
        return avg_policy_loss, metrics
    
    def update_vae(self):
        """Update VAE parameters"""
        vae_loss, metrics = self.compute_vae_loss()
        
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.config.max_grad_norm)
        
        self.vae_optimizer.step()
        
        # Log metrics
        self.logger.log_scalar('train/vae_loss', metrics['vae_loss'], self.step_count)
        self.logger.log_scalar('train/reconstruction_loss', metrics['reconstruction_loss'], self.step_count)
        self.logger.log_scalar('train/kl_loss', metrics['kl_loss'], self.step_count)
        
        # Log gradient norms
        vae_grad_norm = self._compute_grad_norm(self.vae.parameters())
        self.logger.log_scalar('network/vae_grad_norm', vae_grad_norm, self.step_count)
    
    def update_policy(self):
        """Update policy parameters"""
        policy_loss, metrics = self.compute_policy_loss()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        
        self.policy_optimizer.step()
        
        # Log metrics
        self.logger.log_scalar('train/policy_loss', metrics['policy_loss'], self.step_count)
        self.logger.log_scalar('train/avg_return', metrics['avg_return'], self.step_count)
        
        # Log gradient norms
        policy_grad_norm = self._compute_grad_norm(self.policy.parameters())
        self.logger.log_scalar('network/policy_grad_norm', policy_grad_norm, self.step_count)
    
    def train(self, num_episodes: int):
        """Main training loop"""
        logging.info(f"Starting training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            # Sample new task
            self.sample_mdp()
            
            # Collect trajectory
            self.collect_trajectory()
            
            # Update networks
            self.update_vae()
            self.update_policy()
            
            self.step_count += 1
            
            # Periodic logging
            if episode % self.config.log_interval == 0:
                logging.info(f"Episode {episode}/{num_episodes} completed")
        
        logging.info("Training completed")
        self.logger.close()
    
    # Helper methods
    def _get_trajectory_slice(self, trajectory: Dict, t: int) -> Dict:
        """Get trajectory data up to timestep t"""
        return {
            'states': trajectory['states'][:t+1],
            'actions': trajectory['actions'][:t],
            'rewards': trajectory['rewards'][:t]
        }
    
    def _sample_latent(self, posterior) -> torch.Tensor:
        """Sample latent variable using reparameterization trick"""
        # Assumes posterior is a dict with 'mean' and 'logvar'
        mean, logvar = posterior['mean'], posterior['logvar']
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def _compute_returns(self, rewards: List[float]) -> List[float]:
        """Compute discounted returns"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.config.gamma * R
            returns.insert(0, R)
        return returns
    
    def _get_prior(self, t: int):
        """Get prior for timestep t (previous posterior or N(0,I))"""
        if t == 0:
            # Initial prior
            return {'mean': torch.zeros(self.config.latent_dim), 
                   'logvar': torch.zeros(self.config.latent_dim)}
        else:
            # Previous posterior as prior
            traj_slice = self._get_trajectory_slice(self.current_trajectory, t-1)
            return self.vae.encoder(traj_slice)
    
    def _compute_kl_divergence(self, posterior, prior) -> torch.Tensor:
        """Compute KL divergence between two Gaussian distributions"""
        # KL(N(μ1,σ1²) || N(μ2,σ2²)) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2
        mu1, logvar1 = posterior['mean'], posterior['logvar']
        mu2, logvar2 = prior['mean'], prior['logvar']
        
        return 0.5 * torch.sum(
            logvar2 - logvar1 + 
            (torch.exp(logvar1) + (mu1 - mu2)**2) / torch.exp(logvar2) - 1
        )
    
    def _compute_grad_norm(self, parameters) -> float:
        """Compute gradient norm for logging"""
        total_norm = 0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)