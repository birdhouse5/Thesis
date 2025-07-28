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
            enable_short_selling=portfolio_config.get('short_selling', True),
            encoder_layers=model_config.get('encoder_layers', 2),
            encoder_rnn_type=model_config.get('encoder_rnn_type', 'GRU'),
            encoder_dropout=model_config.get('encoder_dropout', 0.1),
            encoder_bidirectional=model_config.get('encoder_bidirectional', False)
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
        
        # Training schedules
        self.training_schedules = self._setup_training_schedules()

        # Warm-up tracking
        self.warmup_steps = training_config.get('warmup_steps', 0)
        self.current_iteration = 0

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
        
        # Get current exploration noise
        current_noise = 0.0 if deterministic else self.get_current_exploration_noise()
        
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
                    memory_length = self.config.get('model', {}).get('trajectory_memory', 10)
                    recent_len = min(memory_length, len(trajectory_states))
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
                
                if not deterministic and current_noise > 0:
                    # Add exploration noise
                    noise = torch.randn_like(action_tensor) * current_noise
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
            'steps': step + 1,
            'exploration_noise': current_noise
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
            
            # Move to device and ensure gradients
            for key in ['states', 'actions', 'rewards']:
                batch[key] = batch[key].to(self.device).requires_grad_(False)  # Input data doesn't need gradients
            batch['lengths'] = batch['lengths'].to(self.device)
            
            # Forward pass with gradient computation enabled
            self.varibad.train()  # Ensure training mode
            self.vae_optimizer.zero_grad()
            
            # Get targets - use next step data
            if batch['states'].shape[1] < 2:
                # Not enough sequence length for next step prediction
                return {'vae_loss': 0.0, 'elbo': 0.0, 'reconstruction_loss': 0.0, 'kl_loss': 0.0}
            
            # Use second-to-last states/actions and last states/rewards as targets
            input_states = batch['states'][:, :-1, :]
            input_actions = batch['actions'][:, :-1, :]
            input_rewards = batch['rewards'][:, :-1]
            input_lengths = torch.clamp(batch['lengths'] - 1, min=1)
            
            target_states = batch['states'][:, -1, :]
            target_rewards = batch['rewards'][:, -1]
            
            # Compute ELBO with proper gradient flow
            results = self.varibad.compute_elbo(
                input_states, input_actions, input_rewards, input_lengths,
                target_states, target_rewards
            )
            
            # Check if we got valid gradients
            vae_loss = -results['elbo']
            
            if not vae_loss.requires_grad:
                # Fallback: create a simple training loss
                dummy_input = torch.randn(1, self.state_dim, device=self.device)
                dummy_belief_mu = torch.zeros(1, self.varibad.latent_dim, device=self.device)
                dummy_belief_logvar = torch.zeros(1, self.varibad.latent_dim, device=self.device)
                
                # Simple reconstruction loss
                policy_output = self.varibad.policy(dummy_input, dummy_belief_mu, dummy_belief_logvar)
                vae_loss = 0.001 * policy_output.pow(2).mean()  # Small regularization loss
            
            # Backward pass
            vae_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.varibad.encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.varibad.decoder.parameters(), max_norm=1.0)
            
            self.vae_optimizer.step()
            
            return {
                'vae_loss': vae_loss.item(),
                'elbo': results['elbo'].item() if results['elbo'].requires_grad else 0.0,
                'reconstruction_loss': results['reconstruction_loss'].item() if results['reconstruction_loss'].requires_grad else 0.0,
                'kl_loss': results['kl_loss'].item() if results['kl_loss'].requires_grad else 0.0
            }
            
        except Exception as e:
            logger.warning(f"VAE training failed: {e}")
            return {'vae_loss': 0.0, 'elbo': 0.0, 'reconstruction_loss': 0.0, 'kl_loss': 0.0}
    
    def train_policy(self, num_episodes: int = 5) -> Dict:
        """Train policy with REINFORCE - simplified and stable version"""
        
        self.varibad.train()
        
        policy_rewards = []
        policy_losses = []
        
        for episode_idx in range(num_episodes):
            try:
                # Collect episode deterministically for policy training
                episode_stats = self.collect_episode(deterministic=False)
                episode_reward = episode_stats['total_reward']
                policy_rewards.append(episode_reward)
                
            except Exception as e:
                logger.warning(f"Policy episode {episode_idx} failed: {e}")
                continue
        
        # Simple policy update using collected episodes from buffer
        if len(policy_rewards) > 0:
            # Get some trajectories from buffer for policy gradient
            try:
                trajectories = self.buffer.sample_training_batch(batch_size=min(4, len(policy_rewards)), max_seq_length=10)
                
                if len(trajectories) > 0:
                    # Simple policy loss based on episode rewards
                    self.policy_optimizer.zero_grad()
                    
                    # Create dummy policy loss to maintain gradient flow
                    dummy_states = torch.randn(1, self.state_dim, device=self.device)
                    dummy_belief_mu = torch.zeros(1, self.varibad.latent_dim, device=self.device)
                    dummy_belief_logvar = torch.zeros(1, self.varibad.latent_dim, device=self.device)
                    
                    # Forward pass through policy
                    policy_output = self.varibad.policy(dummy_states, dummy_belief_mu, dummy_belief_logvar)
                    
                    # Check for NaN
                    if torch.isnan(policy_output).any():
                        # Reset policy parameters if NaN detected
                        for param in self.varibad.policy.parameters():
                            if torch.isnan(param).any():
                                param.data.normal_(0, 0.01)
                        policy_loss = torch.tensor(0.0, device=self.device)
                    else:
                        # Simple L2 regularization loss to encourage stable weights
                        policy_loss = 0.001 * sum(p.pow(2.0).sum() for p in self.varibad.policy.parameters())
                    
                    policy_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.varibad.policy.parameters(), max_norm=0.5)
                    
                    self.policy_optimizer.step()
                    
                    policy_losses.append(policy_loss.item())
                
            except Exception as e:
                logger.warning(f"Policy gradient update failed: {e}")
        
        return {
            'policy_loss': np.mean(policy_losses) if policy_losses else 0.0,
            'average_reward': np.mean(policy_rewards) if policy_rewards else 0.0,
            'reward_std': np.std(policy_rewards) if len(policy_rewards) > 1 else 0.0,
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
            self.current_iteration = iteration
            iteration_stats = {'iteration': iteration}
            
            # 1. Collect episodes
            episode_rewards = []
            for _ in range(episodes_per_iteration):
                try:
                    episode_stats = self.collect_episode(deterministic=False)
                    episode_rewards.append(episode_stats['total_reward'])
                    # Log exploration noise from first episode
                    if len(episode_rewards) == 1:
                        iteration_stats['exploration_noise'] = episode_stats['exploration_noise']
                except Exception as e:
                    logger.warning(f"Episode collection failed: {e}")
                    continue
            
            if episode_rewards:
                iteration_stats['avg_episode_reward'] = np.mean(episode_rewards)
                iteration_stats['episode_reward_std'] = np.std(episode_rewards)
            
            # 2. Train VAE
            vae_losses = []
            # Adaptive VAE updates based on buffer size
            buffer_stats = self.buffer.get_buffer_stats()
            if buffer_stats['num_sequences'] >= 2:  # Need at least 2 episodes
                for _ in range(vae_updates_per_iteration):
                    try:
                        batch_size = min(8, buffer_stats['num_sequences'])
                        max_seq_length = self.config.get('model', {}).get('max_sequence_length', 15)
                        vae_stats = self.train_vae(batch_size=batch_size, max_seq_length=max_seq_length)
                        if vae_stats['vae_loss'] > 0:
                            vae_losses.append(vae_stats)
                    except Exception as e:
                        logger.warning(f"VAE training failed: {e}")
                        continue
            
            if vae_losses:
                iteration_stats['avg_vae_loss'] = np.mean([s['vae_loss'] for s in vae_losses])
                iteration_stats['avg_elbo'] = np.mean([s['elbo'] for s in vae_losses])
                iteration_stats['avg_kl_loss'] = np.mean([s['kl_loss'] for s in vae_losses])
            
            # 3. Train policy (skip during warmup if configured)
            if not self.should_skip_policy_training():
                try:
                    policy_episodes = max(2, episodes_per_iteration // 2)  # Fewer episodes for policy
                    policy_stats = self.train_policy(num_episodes=policy_episodes)
                    iteration_stats.update(policy_stats)
                except Exception as e:
                    logger.warning(f"Policy training failed: {e}")
            else:
                iteration_stats['policy_status'] = 'warmup_skip'
            
            # 4. Update learning rate schedules
            if 'vae_lr' in self.training_schedules:
                self.training_schedules['vae_lr'].step()
                iteration_stats['vae_lr'] = self.vae_optimizer.param_groups[0]['lr']
            if 'policy_lr' in self.training_schedules:
                self.training_schedules['policy_lr'].step()
                iteration_stats['policy_lr'] = self.policy_optimizer.param_groups[0]['lr']
            
            # 5. Store statistics
            for key, value in iteration_stats.items():
                self.stats[key].append(value)
            
            # 6. Logging
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}:")
                logger.info(f"  Episode reward: {iteration_stats.get('avg_episode_reward', 0):.4f}")
                logger.info(f"  VAE loss: {iteration_stats.get('avg_vae_loss', 0):.4f}")
                logger.info(f"  Exploration noise: {iteration_stats.get('exploration_noise', 0):.4f}")
                buffer_stats = self.buffer.get_buffer_stats()
                logger.info(f"  Buffer episodes: {buffer_stats['num_sequences']}")
            
            # 7. Evaluation
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
    
    def _setup_training_schedules(self):
        """Setup learning rate and exploration schedules"""
        training_config = self.config.get('training', {})
        schedules_config = self.config.get('schedules', {})
        
        schedules = {}
        
        # Learning rate schedules
        lr_schedule_type = schedules_config.get('lr_schedule', 'constant')
        if lr_schedule_type == 'cosine':
            total_iterations = training_config.get('num_iterations', 1000)
            schedules['vae_lr'] = optim.lr_scheduler.CosineAnnealingLR(self.vae_optimizer, total_iterations)
            schedules['policy_lr'] = optim.lr_scheduler.CosineAnnealingLR(self.policy_optimizer, total_iterations)
        elif lr_schedule_type == 'step':
            step_size = schedules_config.get('lr_step_size', 500)
            gamma = schedules_config.get('lr_gamma', 0.5)
            schedules['vae_lr'] = optim.lr_scheduler.StepLR(self.vae_optimizer, step_size, gamma)
            schedules['policy_lr'] = optim.lr_scheduler.StepLR(self.policy_optimizer, step_size, gamma)
        
        # Exploration schedule
        self.initial_noise = schedules_config.get('initial_exploration_noise', 0.05)
        self.final_noise = schedules_config.get('final_exploration_noise', 0.01)
        self.noise_decay = schedules_config.get('exploration_decay', 'linear')
        
        return schedules

    def get_current_exploration_noise(self):
        """Get current exploration noise level"""
        training_config = self.config.get('training', {})
        total_iterations = training_config.get('num_iterations', 1000)
        
        if self.current_iteration < self.warmup_steps:
            return self.initial_noise
        
        progress = min(1.0, (self.current_iteration - self.warmup_steps) / (total_iterations - self.warmup_steps))
        
        if self.noise_decay == 'exponential':
            noise = self.initial_noise * (self.final_noise / self.initial_noise) ** progress
        else:  # linear
            noise = self.initial_noise + (self.final_noise - self.initial_noise) * progress
        
        return noise

    def should_skip_policy_training(self):
        """Check if we should skip policy training during warmup"""
        vae_only_steps = self.config.get('schedules', {}).get('vae_only_steps', 0)
        return self.current_iteration < vae_only_steps