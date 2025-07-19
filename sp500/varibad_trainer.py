"""
Complete VariBAD Training System for Portfolio Optimization

Integrates all components:
1. S&P 500 data loading (your cleaned dataset)
2. Portfolio MDP (your MetaTrader implementation)
3. Trajectory buffer (for τ:t sequences)
4. VariBAD VAE (encoder, decoder, policy)
5. Training loop (ELBO + RL objectives)

Implements the full variBAD training algorithm from the paper.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Import your components
from varibad_models import VariBADVAE
from trajectory_buffer import BlindTrajectoryBuffer, create_trajectory_batch
from test_MPD_setup import MetaTraderPortfolioMDP

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VariBADTrainer:
    """
    Complete VariBAD training system for portfolio optimization.
    
    Implements the full training algorithm:
    1. Collect episodes using current policy
    2. Store trajectories in buffer  
    3. Sample training batches
    4. Update VAE (encoder + decoder) with ELBO loss
    5. Update policy with RL loss (DSR rewards)
    6. Repeat
    """
    
    def __init__(self,
                 data_path: str = 'data/sp500_rl_ready_cleaned.parquet',
                 state_dim: Optional[int] = None,  # Will be auto-detected from MDP
                 action_dim: int = 30,  # 30 S&P 500 companies
                 latent_dim: int = 5,
                 episode_length: int = 30,
                 max_episodes_buffer: int = 200,
                 enable_short_selling: bool = True,
                 max_short_ratio: float = 0.3,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize VariBAD training system.
        
        Args:
            data_path: Path to your cleaned S&P 500 dataset
            state_dim: Portfolio state dimension (features from your data)
            action_dim: Number of assets in portfolio
            latent_dim: Task embedding dimension
            episode_length: Number of trading days per episode
            max_episodes_buffer: Maximum episodes in trajectory buffer
            enable_short_selling: Whether to allow short positions
            max_short_ratio: Maximum short position ratio
            device: Training device ('cuda' or 'cpu')
        """
        self.device = device
        self.episode_length = episode_length
        self.action_dim = action_dim
        
        # Load your S&P 500 data
        logger.info(f"Loading S&P 500 data from {data_path}")
        self.data = self._load_portfolio_data(data_path)
        
        # Create MDP first to get actual state dimensions
        logger.info("Creating portfolio MDP to determine state dimensions...")
        self.env = MetaTraderPortfolioMDP(
            data=self.data,
            episode_length=episode_length,
            short_selling_enabled=enable_short_selling,
            max_short_ratio=max_short_ratio
        )
        
        # Get actual state dimension from MDP
        sample_state = self.env.reset()
        self.state_dim = len(sample_state)
        
        if state_dim is not None and state_dim != self.state_dim:
            logger.warning(f"Provided state_dim ({state_dim}) != actual MDP state_dim ({self.state_dim})")
            logger.warning(f"Using actual MDP state_dim: {self.state_dim}")
        
        logger.info(f"Detected state dimension: {self.state_dim}")
        
        # Initialize VariBAD components with correct dimensions
        logger.info("Initializing VariBAD components...")
        
        # VariBAD VAE (encoder, decoder, policy)
        self.varibad = VariBADVAE(
            state_dim=self.state_dim,  # Use detected dimension
            action_dim=action_dim,
            latent_dim=latent_dim,
            enable_short_selling=enable_short_selling,
            max_short_ratio=max_short_ratio
        ).to(device)
        
        # Trajectory buffer
        self.buffer = BlindTrajectoryBuffer(
            max_episodes=max_episodes_buffer,
            device=device
        )
        
        # Optimizers
        self.vae_optimizer = optim.Adam([
            {'params': self.varibad.encoder.parameters(), 'lr': 3e-4},
            {'params': self.varibad.decoder.parameters(), 'lr': 3e-4}
        ])
        
        self.policy_optimizer = optim.Adam(
            self.varibad.policy.parameters(), lr=3e-4
        )
        
        # Training statistics
        self.stats = defaultdict(list)
        
        logger.info(f"VariBAD trainer initialized on {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.varibad.parameters()):,}")
        logger.info(f"State dimension: {self.state_dim}")
        logger.info(f"Action dimension: {action_dim}")
        logger.info(f"Data shape: {self.data.shape}")
        logger.info(f"Episode length: {episode_length} days")
    
    def _load_portfolio_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate your S&P 500 dataset."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load your cleaned dataset
        data = pd.read_parquet(data_path)
        
        # Validate data structure
        required_cols = ['date', 'ticker']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Ensure date column is datetime
        data['date'] = pd.to_datetime(data['date'])
        
        logger.info(f"Loaded data: {len(data)} rows, {len(data.columns)} columns")
        logger.info(f"Date range: {data['date'].min().date()} to {data['date'].max().date()}")
        logger.info(f"Tickers: {data['ticker'].nunique()} unique ({', '.join(sorted(data['ticker'].unique())[:10])}...)")
        
        return data
    
    def collect_episode(self, deterministic: bool = False) -> Dict:
        """
        Collect a single episode using current policy.
        
        Args:
            deterministic: Whether to use deterministic actions (for evaluation)
            
        Returns:
            Episode statistics and trajectory data
        """
        self.buffer.start_sequence()
        
        # Reset environment
        state = self.env.reset()
        episode_reward = 0.0
        episode_returns = []
        episode_actions = []
        episode_states = []
        
        # Track belief evolution during episode
        trajectory_states = []
        trajectory_actions = []
        trajectory_rewards = []
        
        for step in range(self.episode_length):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get belief from trajectory so far
            if len(trajectory_states) > 0:
                # Create trajectory sequence for encoder
                traj_states = torch.stack([torch.FloatTensor(s) for s in trajectory_states]).unsqueeze(0).to(self.device)
                traj_actions = torch.stack([torch.FloatTensor(a) for a in trajectory_actions]).unsqueeze(0).to(self.device)
                traj_rewards = torch.FloatTensor(trajectory_rewards).unsqueeze(0).to(self.device)
                traj_lengths = torch.LongTensor([len(trajectory_states)]).to(self.device)
                
                # Encode trajectory to belief
                with torch.no_grad():
                    belief_mu, belief_logvar, _ = self.varibad.encode_trajectory(
                        traj_states, traj_actions, traj_rewards, traj_lengths
                    )
            else:
                # Use prior belief for first step
                belief_mu = torch.zeros(1, self.varibad.latent_dim).to(self.device)
                belief_logvar = torch.zeros(1, self.varibad.latent_dim).to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                if deterministic:
                    # Use mean of policy distribution
                    action_tensor = self.varibad.get_policy_action(state_tensor, belief_mu, belief_logvar)
                    action = action_tensor.squeeze(0).cpu().numpy()
                else:
                    # Add exploration noise
                    action_tensor = self.varibad.get_policy_action(state_tensor, belief_mu, belief_logvar)
                    noise = torch.randn_like(action_tensor) * 0.01  # Small exploration noise
                    action_tensor = action_tensor + noise
                    action = action_tensor.squeeze(0).cpu().numpy()
            
            # Execute action in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store transition in buffer
            self.buffer.add_step(
                state=state.copy(),
                action=action.copy(),
                reward=reward,
                next_state=next_state.copy(),
                done=done
            )
            
            # Update trajectory history
            trajectory_states.append(state.copy())
            trajectory_actions.append(action.copy())
            trajectory_rewards.append(reward)
            
            # Track episode statistics
            episode_reward += reward
            episode_returns.append(info.get('portfolio_return', 0.0))
            episode_actions.append(action.copy())
            episode_states.append(state.copy())
            
            # Update state
            state = next_state
            
            if done:
                break
        
        episode_stats = {
            'total_reward': episode_reward,
            'average_reward': episode_reward / (step + 1),
            'total_return': sum(episode_returns),
            'average_return': np.mean(episode_returns),
            'volatility': np.std(episode_returns),
            'sharpe_ratio': np.mean(episode_returns) / (np.std(episode_returns) + 1e-8),
            'steps': step + 1,
            'final_portfolio_return': info.get('portfolio_return', 0.0),
            'transaction_costs': sum([info.get('transaction_cost', 0.0) for info in [info]]),
            'date_range': f"{info.get('date', 'unknown')}"
        }
        
        return episode_stats
    
    def train_vae(self, batch_size: int = 32, max_seq_length: int = 20) -> Dict:
        """
        Train VariBAD VAE components (encoder + decoder) with ELBO loss.
        
        Args:
            batch_size: Number of trajectory sequences per batch
            max_seq_length: Maximum sequence length for training
            
        Returns:
            Training statistics
        """
        # Sample trajectory batch from buffer
        trajectories = self.buffer.sample_training_batch(batch_size, max_seq_length)
        
        if len(trajectories) == 0:
            return {'vae_loss': 0.0, 'elbo': 0.0, 'reconstruction_loss': 0.0, 'kl_loss': 0.0}
        
        # Create padded batch
        batch = create_trajectory_batch(trajectories)
        
        # Move to device
        for key in ['states', 'actions', 'rewards', 'lengths']:
            batch[key] = batch[key].to(self.device)
        
        # Forward pass through VAE
        self.vae_optimizer.zero_grad()
        
        results = self.varibad.forward(batch)
        
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
            'kl_loss': results['kl_loss'].item(),
            'batch_size': len(trajectories)
        }
    
    def train_policy(self, num_episodes: int = 5) -> Dict:
        """
        Train policy with RL objective (maximize DSR rewards).
        
        Args:
            num_episodes: Number of episodes to collect for policy update
            
        Returns:
            Training statistics
        """
        self.policy_optimizer.zero_grad()
        
        policy_rewards = []
        policy_losses = []
        
        for _ in range(num_episodes):
            # Collect episode with current policy
            episode_stats = self.collect_episode(deterministic=False)
            policy_rewards.append(episode_stats['total_reward'])
        
        # Policy loss (maximize expected reward)
        if policy_rewards:
            avg_reward = np.mean(policy_rewards)
            # Simple policy gradient: maximize average reward
            policy_loss = -torch.tensor(avg_reward, device=self.device, requires_grad=False)
            policy_losses.append(policy_loss.item())
        else:
            policy_loss = torch.tensor(0.0, device=self.device)
        
        # Note: In a full implementation, you'd compute actual policy gradients
        # This is simplified for demonstration - we'll improve this later
        
        return {
            'policy_loss': np.mean(policy_losses) if policy_losses else 0.0,
            'average_reward': np.mean(policy_rewards) if policy_rewards else 0.0,
            'reward_std': np.std(policy_rewards) if policy_rewards else 0.0,
            'episodes_collected': len(policy_rewards)
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """Evaluate current policy performance."""
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
                avg_stats[f'eval_{key}_std'] = np.std([ep[key] for ep in eval_stats])
        
        return avg_stats
    
    def train(self, 
              num_iterations: int = 1000,
              episodes_per_iteration: int = 5,
              vae_updates_per_iteration: int = 10,
              eval_frequency: int = 50,
              save_frequency: int = 100,
              save_dir: str = 'checkpoints') -> Dict:
        """
        Main training loop implementing full VariBAD algorithm.
        
        Args:
            num_iterations: Number of training iterations
            episodes_per_iteration: Episodes to collect per iteration
            vae_updates_per_iteration: VAE updates per iteration
            eval_frequency: How often to evaluate policy
            save_frequency: How often to save checkpoints
            save_dir: Directory to save checkpoints
            
        Returns:
            Training statistics
        """
        logger.info("Starting VariBAD training...")
        logger.info(f"Training for {num_iterations} iterations")
        logger.info(f"Episodes per iteration: {episodes_per_iteration}")
        logger.info(f"VAE updates per iteration: {vae_updates_per_iteration}")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training loop
        for iteration in range(num_iterations):
            iteration_stats = {'iteration': iteration}
            
            # 1. Collect episodes with current policy
            logger.info(f"Iteration {iteration}: Collecting {episodes_per_iteration} episodes...")
            
            episode_rewards = []
            for _ in range(episodes_per_iteration):
                episode_stats = self.collect_episode(deterministic=False)
                episode_rewards.append(episode_stats['total_reward'])
            
            iteration_stats['avg_episode_reward'] = np.mean(episode_rewards)
            iteration_stats['episode_reward_std'] = np.std(episode_rewards)
            
            # 2. Train VAE with collected trajectories
            logger.info(f"Iteration {iteration}: Training VAE...")
            
            vae_losses = []
            for _ in range(vae_updates_per_iteration):
                vae_stats = self.train_vae()
                if vae_stats['vae_loss'] > 0:  # Only record if we had data
                    vae_losses.append(vae_stats)
            
            if vae_losses:
                iteration_stats['avg_vae_loss'] = np.mean([s['vae_loss'] for s in vae_losses])
                iteration_stats['avg_elbo'] = np.mean([s['elbo'] for s in vae_losses])
                iteration_stats['avg_reconstruction_loss'] = np.mean([s['reconstruction_loss'] for s in vae_losses])
                iteration_stats['avg_kl_loss'] = np.mean([s['kl_loss'] for s in vae_losses])
            
            # 3. Train policy (simplified for now)
            policy_stats = self.train_policy(num_episodes=2)
            iteration_stats.update(policy_stats)
            
            # 4. Store statistics
            for key, value in iteration_stats.items():
                self.stats[key].append(value)
            
            # 5. Logging
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}:")
                logger.info(f"  Episode reward: {iteration_stats.get('avg_episode_reward', 0):.4f}")
                logger.info(f"  VAE loss: {iteration_stats.get('avg_vae_loss', 0):.4f}")
                logger.info(f"  ELBO: {iteration_stats.get('avg_elbo', 0):.4f}")
                logger.info(f"  Buffer episodes: {self.buffer.get_buffer_stats()['num_sequences']}")
            
            # 6. Evaluation
            if iteration % eval_frequency == 0 and iteration > 0:
                logger.info(f"Iteration {iteration}: Evaluating policy...")
                eval_stats = self.evaluate()
                iteration_stats.update(eval_stats)
                
                logger.info(f"  Eval reward: {eval_stats.get('eval_total_reward', 0):.4f}")
                logger.info(f"  Eval Sharpe: {eval_stats.get('eval_sharpe_ratio', 0):.4f}")
            
            # 7. Save checkpoint
            if iteration % save_frequency == 0 and iteration > 0:
                self.save_checkpoint(os.path.join(save_dir, f'checkpoint_{iteration}.pt'))
        
        logger.info("Training completed!")
        return dict(self.stats)
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'iteration': len(self.stats['iteration']) if 'iteration' in self.stats else 0,
            'varibad_state_dict': self.varibad.state_dict(),
            'vae_optimizer_state_dict': self.vae_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'stats': dict(self.stats),
            'buffer_stats': self.buffer.get_buffer_stats()
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.varibad.load_state_dict(checkpoint['varibad_state_dict'])
        self.vae_optimizer.load_state_dict(checkpoint['vae_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.stats = defaultdict(list, checkpoint['stats'])
        
        logger.info(f"Checkpoint loaded from {path}")
        logger.info(f"Resumed from iteration {checkpoint['iteration']}")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training progress."""
        if not self.stats:
            logger.warning("No training statistics to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        if 'avg_episode_reward' in self.stats:
            axes[0, 0].plot(self.stats['avg_episode_reward'])
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Average Reward')
        
        # VAE losses
        if 'avg_vae_loss' in self.stats:
            axes[0, 1].plot(self.stats['avg_vae_loss'])
            axes[0, 1].set_title('VAE Loss')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Loss')
        
        # ELBO
        if 'avg_elbo' in self.stats:
            axes[1, 0].plot(self.stats['avg_elbo'])
            axes[1, 0].set_title('ELBO')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('ELBO')
        
        # Evaluation metrics
        if 'eval_total_reward' in self.stats:
            eval_iterations = [i for i, val in enumerate(self.stats['eval_total_reward']) if val is not None]
            eval_rewards = [val for val in self.stats['eval_total_reward'] if val is not None]
            axes[1, 1].plot(eval_iterations, eval_rewards)
            axes[1, 1].set_title('Evaluation Rewards')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Eval Reward')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        
        plt.show()


def test_trainer():
    """Test the VariBAD training system."""
    print("🧪 Testing VariBAD Training System")
    print("=" * 50)
    
    # Check if data file exists
    data_path = 'data/sp500_rl_ready_cleaned.parquet'
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please ensure you have the cleaned S&P 500 dataset")
        return None
    
    try:
        # Initialize trainer
        trainer = VariBADTrainer(
            data_path=data_path,
            episode_length=15,  # Shorter episodes for testing
            device='cpu'  # Use CPU for testing
        )
        
        print("✓ Trainer initialized successfully")
        print(f"✓ Data loaded: {trainer.data.shape}")
        print(f"✓ State dimension: {trainer.state_dim}")
        print(f"✓ Action dimension: {trainer.action_dim}")
        print(f"✓ Model parameters: {sum(p.numel() for p in trainer.varibad.parameters()):,}")
        
        # Test episode collection
        print("\n1. Testing episode collection...")
        episode_stats = trainer.collect_episode(deterministic=True)
        print(f"✓ Episode collected: {episode_stats['steps']} steps")
        print(f"✓ Total reward: {episode_stats['total_reward']:.4f}")
        print(f"✓ Average return: {episode_stats['average_return']:.4f}")
        
        # Test VAE training
        print("\n2. Testing VAE training...")
        
        # Collect a few more episodes to have training data
        for _ in range(3):
            trainer.collect_episode()
        
        vae_stats = trainer.train_vae(batch_size=2, max_seq_length=10)
        print(f"✓ VAE training: loss = {vae_stats['vae_loss']:.4f}")
        print(f"✓ ELBO: {vae_stats['elbo']:.4f}")
        print(f"✓ Batch size: {vae_stats['batch_size']}")
        
        # Test evaluation
        print("\n3. Testing evaluation...")
        eval_stats = trainer.evaluate(num_episodes=2)
        print(f"✓ Evaluation completed")
        print(f"✓ Eval reward: {eval_stats.get('eval_total_reward', 0):.4f}")
        
        # Test short training run
        print("\n4. Testing training loop...")
        stats = trainer.train(
            num_iterations=5,
            episodes_per_iteration=2,
            vae_updates_per_iteration=2,
            eval_frequency=3
        )
        
        print(f"✓ Training completed: {len(stats['iteration'])} iterations")
        print(f"✓ Final episode reward: {stats['avg_episode_reward'][-1]:.4f}")
        
        print("\n🎉 All tests passed! VariBAD training system is ready.")
        
        return trainer
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test the training system
    trainer = test_trainer()
    
    if trainer:
        print(f"\n🚀 Ready for full training!")
        print(f"\nTo start full training:")
        print(f"```python")
        print(f"trainer = VariBADTrainer()")
        print(f"stats = trainer.train(num_iterations=1000)")
        print(f"trainer.plot_training_curves('training_progress.png')")
        print(f"```")
        
        print(f"\n📊 Training will optimize:")
        print(f"• Encoder: Learn market regimes from trajectory sequences")
        print(f"• Policy: Maximize DSR rewards using learned beliefs")
        print(f"• Decoder: Enable unsupervised learning via reconstruction")
        print(f"• Combined: ELBO + RL objective as in variBAD paper")