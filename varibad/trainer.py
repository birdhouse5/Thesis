"""
VariBAD training system
Consolidated from varibad/core/trainer.py
"""

import torch
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, Optional
from collections import defaultdict

from .models import VariBADVAE, PortfolioEnvironment
from .utils import TrajectoryBuffer, create_trajectory_batch, get_device, count_parameters
from .data import load_dataset

logger = logging.getLogger(__name__)


class VariBADTrainer:
    """VariBAD training system"""
    
    def __init__(self, config: Dict):
        """Initialize trainer from config"""
        
        self.config = config
        
        # Extract config sections
        training_config = config.get('training', {})
        model_config = config.get('model', {})
        portfolio_config = config.get('portfolio', {})
        env_config = config.get('environment', {})
        
        # Setup device
        self.device = get_device(env_config.get('device', 'auto'))
        logger.info(f"Using device: {self.device}")
        
        # Load data
        data_path = env_config.get('data_path', 'data/sp500_rl_ready_cleaned.parquet')
        self.data = load_dataset(data_path)
        
        # Create environment
        self.env = PortfolioEnvironment(
            data=self.data,
            episode_length=training_config.get('episode_length', 30),
            enable_short_selling=portfolio_config.get('short_selling', True),
            max_short_ratio=portfolio_config.get('max_short_ratio', 0.3),
            transaction_cost=portfolio_config.get('transaction_cost', 0.001)
        )
        
        # Get dimensions from environment
        sample_state = self.env.reset()
        self.state_dim = len(sample_state)
        self.action_dim = self.env.action_space.shape[0]
        
        logger.info(f"Environment setup:")
        logger.info(f"  State dimension: {self.state_dim}")
        logger.info(f"  Action dimension: {self.action_dim}")
        logger.info(f"  Assets: {self.env.n_assets}")
        logger.info(f"  Short selling: {self.env.enable_short_selling}")
        
        # Create VariBAD model
        self.varibad = VariBADVAE(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            latent_dim=model_config.get('latent_dim', 5),
            encoder_hidden=model_config.get('encoder_hidden', 128),
            decoder_hidden=model_config.get('decoder_hidden', 128),
            policy_hidden=model_config.get('policy_hidden', 256),
            enable_short_selling=portfolio_config.get('short_selling', True)
        ).to(self.device)
        
        # Trajectory buffer
        self.buffer = TrajectoryBuffer(
            max_episodes=training_config.get('buffer_size', 200),
            device=self.device
        )
        
        # Optimizers
        learning_rates = config.get('learning_rates', {})
        self.vae_optimizer = optim.Adam([
            {'params': self.varibad.encoder.parameters(), 'lr': learning_rates.get('vae_encoder_lr', 1e-4)},
            {'params': self.varibad.decoder.parameters(), 'lr': learning_rates.get('vae_decoder_lr', 1e-4)}
        ])
        
        self.policy_optimizer = optim.Adam(
            self.varibad.policy.parameters(), 
            lr=learning_rates.get('policy_lr', 1e-4)
        )
        
        # Training statistics
        self.stats = defaultdict(list)
        
        logger.info(f"Model initialized:")
        logger.info(f"  Parameters: {count_parameters(self.varibad):,}")
        logger.info(f"  VAE encoder LR: {learning_rates.get('vae_encoder_lr', 1e-4)}")
        logger.info(f"  VAE decoder LR: {learning_rates.get('vae_decoder_lr', 1e-4)}")
        logger.info(f"  Policy LR: {learning_rates.get('policy_lr', 1e-4)}")
    
    def get_model_info(self) -> str:
        """Get model information string"""
        return f"{count_parameters(self.varibad):,} parameters on {self.device}"
    
    def get_parameter_count(self) -> int:
        """Get total parameter count"""
        return count_parameters(self.varibad)
    
    def collect_episode(self, deterministic: bool = False) -> Dict:
        """Collect a single episode"""
        
        self.buffer.start_sequence()
        
        state = self.env.reset()
        episode_reward = 0.0
        episode_returns = []
        
        # Trajectory for belief computation
        trajectory_states = []
        trajectory_actions = []
        trajectory_rewards = []
        
        for step in range(self.env.episode_length):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get belief from trajectory
            if len(trajectory_states) > 0:
                try:
                    # Use recent trajectory for belief
                    recent_len = min(10, len(trajectory_states))
                    traj_states = torch.stack([torch.FloatTensor(s) for s in trajectory_states[-recent_len:]]).unsqueeze(0).to(self.device)
                    traj_actions = torch.stack([torch.FloatTensor(a) for a in trajectory_actions[-recent_len:]]).unsqueeze(0).to(self.device)
                    traj_rewards = torch.FloatTensor(trajectory_rewards[-recent_len:]).unsqueeze(0).to(self.device)
                    traj_lengths = torch.LongTensor([recent_len]).to(self.device)
                    
                    with torch.no_grad():
                        belief_mu, belief_logvar, _ = self.varibad.encode_trajectory(
                            traj_states, traj_actions, traj_rewards, traj_lengths
                        )
                except:
                    # Fallback to prior
                    belief_mu = torch.zeros(1, self.varibad.latent_dim).to(self.device)
                    belief_logvar = torch.zeros(1, self.varibad.latent_dim).to(self.device)
            else:
                # Prior belief for first step
                belief_mu = torch.zeros(1, self.varibad.latent_dim).to(self.device)
                belief_logvar = torch.zeros(1, self.varibad.latent_dim).to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                action_tensor = self.varibad.get_policy_action(state_tensor, belief_mu, belief_logvar)
                
                if not deterministic:
                    # Add exploration noise
                    noise = torch.randn_like(action_tensor) * 0.01
                    action_tensor = action_tensor + noise
                
                action = action_tensor.squeeze(0).cpu().numpy()
            
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            
            # Store in buffer
            self.buffer.add_step(state, action, reward, next_state, done)
            
            # Update trajectory
            trajectory_states.append(state.copy())
            trajectory_actions.append(action.copy())
            trajectory_rewards.append(reward)
            
            # Track episode stats
            episode_reward += reward
            episode_returns.append(info.get('portfolio_return', 0.0))
            
            state = next_state
            
            if done:
                break
        
        episode_stats = {
            'total_reward': episode_reward,
            'average_reward': episode_reward / (step + 1),
            'total_return': sum(episode_returns),
            'average_return': np.mean(episode_returns),
            'volatility': np.std(episode_returns),
            'steps': step + 1
        }
        
        return episode_stats
    
    def train_vae(self, batch_size: int = 32, max_seq_length: int = 20) -> Dict:
        """Train VAE components"""
        
        # Sample trajectory batch
        trajectories = self.buffer.sample_training_batch(batch_size, max_seq_length)
        
        if len(trajectories) == 0:
            return {'vae_loss': 0.0, 'elbo': 0.0, 'reconstruction_loss': 0.0, 'kl_loss': 0.0}
        
        try:
            # Create padded batch
            batch = create_trajectory_batch(trajectories)
            
            # Move to device
            for key in ['states', 'actions', 'rewards', 'lengths']:
                batch[key] = batch[key].to(self.device)
            
            # Forward pass
            self.vae_optimizer.zero_grad()
            
            # Use last state/reward as targets
            next_states = batch['states'][:, -1, :]
            next_rewards = batch['rewards'][:, -1]
            
            results = self.varibad.compute_elbo(
                batch['states'], batch['actions'], batch['rewards'], batch['lengths'],
                next_states, next_rewards
            )
            
            # VAE loss (minimize negative ELBO)
            vae_loss = -results['elbo']
            
            # Backward pass
            vae_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.varibad.encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.varibad.decoder.parameters(), max_norm=1.0)
            
            self.vae_optimizer.step()
            
            return {
                'vae_loss': vae_loss.item(),
                'elbo': results['elbo'].item(),
                'reconstruction_loss': results['reconstruction_loss'].item(),
                'kl_loss': results['kl_loss'].item()
            }
            
        except Exception as e:
            logger.warning(f"VAE training failed: {e}")
            return {'vae_loss': 0.0, 'elbo': 0.0, 'reconstruction_loss': 0.0, 'kl_loss': 0.0}
    
    def train_policy(self, num_episodes: int = 5) -> Dict:
        """Train policy with REINFORCE"""
        
        self.varibad.train()
        
        episode_trajectories = []
        policy_rewards = []
        
        for _ in range(num_episodes):
            # Collect episode with gradient tracking
            trajectory = {
                'states': [],
                'actions': [],
                'rewards': [],
                'log_probs': []
            }
            
            state = self.env.reset()
            episode_reward = 0.0
            episode_states = []
            episode_actions = []
            episode_rewards = []
            
            for step in range(self.env.episode_length):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # Get belief (simplified)
                if len(episode_states) > 0:
                    try:
                        recent_states = torch.stack([torch.FloatTensor(s) for s in episode_states[-5:]]).unsqueeze(0).to(self.device)
                        recent_actions = torch.stack([torch.FloatTensor(a) for a in episode_actions[-5:]]).unsqueeze(0).to(self.device)
                        recent_rewards = torch.FloatTensor(episode_rewards[-5:]).unsqueeze(0).to(self.device)
                        recent_lengths = torch.LongTensor([len(recent_states[0])]).to(self.device)
                        
                        belief_mu, belief_logvar, _ = self.varibad.encode_trajectory(
                            recent_states, recent_actions, recent_rewards, recent_lengths
                        )
                    except:
                        belief_mu = torch.zeros(1, self.varibad.latent_dim).to(self.device)
                        belief_logvar = torch.zeros(1, self.varibad.latent_dim).to(self.device)
                else:
                    belief_mu = torch.zeros(1, self.varibad.latent_dim).to(self.device)
                    belief_logvar = torch.zeros(1, self.varibad.latent_dim).to(self.device)
                
                # Get action with gradient tracking
                action_probs = self.varibad.policy(state_tensor, belief_mu, belief_logvar)
                
                # Create distribution and sample
                if self.env.enable_short_selling:
                    # Simplified: use softmax over all weights
                    dist = torch.distributions.Categorical(logits=action_probs.flatten())
                    action_idx = dist.sample()
                    
                    action = torch.zeros_like(action_probs)
                    action.flatten()[action_idx] = 1.0
                    log_prob = dist.log_prob(action_idx)
                else:
                    dist = torch.distributions.Categorical(logits=action_probs)
                    action_idx = dist.sample()
                    
                    action = torch.zeros_like(action_probs)
                    action[0, action_idx] = 1.0
                    log_prob = dist.log_prob(action_idx)
                
                # Execute action
                action_np = action.detach().cpu().numpy().flatten()
                next_state, reward, done, info = self.env.step(action_np)
                
                # Store trajectory
                trajectory['states'].append(state_tensor.squeeze(0))
                trajectory['actions'].append(action.squeeze(0))
                trajectory['rewards'].append(reward)
                trajectory['log_probs'].append(log_prob)
                
                # Update for next step
                episode_states.append(state.copy())
                episode_actions.append(action_np.copy())
                episode_rewards.append(reward)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_trajectories.append(trajectory)
            policy_rewards.append(episode_reward)
        
        # Policy gradient update
        self.policy_optimizer.zero_grad()
        
        total_policy_loss = 0.0
        
        for trajectory in episode_trajectories:
            # Calculate returns
            rewards = trajectory['rewards']
            returns = []
            G = 0
            
            for r in reversed(rewards):
                G = r + 0.99 * G
                returns.insert(0, G)
            
            returns = torch.FloatTensor(returns).to(self.device)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Policy loss
            log_probs = torch.stack(trajectory['log_probs'])
            policy_loss = -(log_probs * returns).sum()
            total_policy_loss += policy_loss
        
        # Average loss
        avg_policy_loss = total_policy_loss / len(episode_trajectories)
        
        # Backward pass
        avg_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.varibad.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()
        
        return {
            'policy_loss': avg_policy_loss.item(),
            'average_reward': np.mean(policy_rewards),
            'reward_std': np.std(policy_rewards),
            'episodes_collected': len(policy_rewards)
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """Evaluate current policy"""
        
        self.varibad.eval()
        
        eval_stats = []
        
        with torch.no_grad():
            for _ in range(num_episodes):
                episode_stats = self.collect_episode(deterministic=True)
                eval_stats.append(episode_stats)
        
        self.varibad.train()
        
        # Aggregate statistics
        avg_stats = {}
        for key in eval_stats[0].keys():
            if isinstance(eval_stats[0][key], (int, float)):
                avg_stats[f'eval_{key}'] = np.mean([ep[key] for ep in eval_stats])
        
        return avg_stats
    
    def train(self) -> Dict:
        """Main training loop"""
        
        training_config = self.config.get('training', {})
        
        num_iterations = training_config.get('num_iterations', 1000)
        episodes_per_iteration = training_config.get('episodes_per_iteration', 5)
        vae_updates_per_iteration = training_config.get('vae_updates', 10)
        eval_frequency = training_config.get('eval_frequency', 50)
        
        logger.info("Starting VariBAD training...")
        logger.info(f"Training for {num_iterations} iterations")
        
        for iteration in range(num_iterations):
            iteration_stats = {'iteration': iteration}
            
            # 1. Collect episodes
            episode_rewards = []
            for _ in range(episodes_per_iteration):
                try:
                    episode_stats = self.collect_episode(deterministic=False)
                    episode_rewards.append(episode_stats['total_reward'])
                except Exception as e:
                    logger.warning(f"Episode collection failed: {e}")
                    continue
            
            if episode_rewards:
                iteration_stats['avg_episode_reward'] = np.mean(episode_rewards)
                iteration_stats['episode_reward_std'] = np.std(episode_rewards)
            
            # 2. Train VAE
            vae_losses = []
            for _ in range(vae_updates_per_iteration):
                try:
                    vae_stats = self.train_vae(batch_size=8, max_seq_length=15)
                    if vae_stats['vae_loss'] > 0:
                        vae_losses.append(vae_stats)
                except Exception as e:
                    logger.warning(f"VAE training failed: {e}")
                    continue
            
            if vae_losses:
                iteration_stats['avg_vae_loss'] = np.mean([s['vae_loss'] for s in vae_losses])
                iteration_stats['avg_elbo'] = np.mean([s['elbo'] for s in vae_losses])
            
            # 3. Train policy
            try:
                policy_stats = self.train_policy(num_episodes=3)
                iteration_stats.update(policy_stats)
            except Exception as e:
                logger.warning(f"Policy training failed: {e}")
            
            # 4. Store statistics
            for key, value in iteration_stats.items():
                self.stats[key].append(value)
            
            # 5. Logging
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}:")
                logger.info(f"  Episode reward: {iteration_stats.get('avg_episode_reward', 0):.4f}")
                logger.info(f"  VAE loss: {iteration_stats.get('avg_vae_loss', 0):.4f}")
                buffer_stats = self.buffer.get_buffer_stats()
                logger.info(f"  Buffer episodes: {buffer_stats['num_sequences']}")
            
            # 6. Evaluation
            if iteration % eval_frequency == 0 and iteration > 0:
                try:
                    eval_stats = self.evaluate(num_episodes=5)
                    iteration_stats.update(eval_stats)
                    logger.info(f"Evaluation - Avg reward: {eval_stats.get('eval_total_reward', 0):.4f}")
                except Exception as e:
                    logger.warning(f"Evaluation failed: {e}")
        
        logger.info("Training completed!")
        return dict(self.stats)
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        
        checkpoint = {
            'iteration': len(self.stats.get('iteration', [])),
            'model_state_dict': self.varibad.state_dict(),
            'vae_optimizer_state_dict': self.vae_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'training_stats': dict(self.stats),
            'config': self.config,
            'model_info': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'parameter_count': count_parameters(self.varibad),
                'device': str(self.device)
            }
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.varibad.load_state_dict(checkpoint['model_state_dict'])
        self.vae_optimizer.load_state_dict(checkpoint['vae_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.stats = defaultdict(list, checkpoint.get('training_stats', {}))
        
        logger.info(f"Checkpoint loaded: {path}")
        logger.info(f"Resumed from iteration: {checkpoint.get('iteration', 0)}")
        
        return checkpoint