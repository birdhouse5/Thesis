# File: main.py
import torch
import logging
import numpy as np
from pathlib import Path

from logger_config import setup_experiment_logging
from environments.data_preparation import load_dataset
from environments.dataset import Dataset
from environments.env import MetaEnv
from models.vae import VAE
from models.policy import PortfolioPolicy
from algorithms.trainer import PPOTrainer

logger = logging.getLogger(__name__)

class Config:
    """Training configuration for VariBAD Portfolio Optimization"""
    def __init__(self):
        # Environment parameters
        self.data_path = "environments/data/sp500_rl_ready_cleaned_fixed.parquet"
        self.num_assets = 30
        self.seq_len = 60           # Fixed episode length (task sequence length)
        self.min_horizon = 45       # Minimum episode length within task 
        self.max_horizon = 60       # Maximum episode length within task
        
        # Model parameters
        self.latent_dim = 64
        self.hidden_dim = 256
        self.vae_lr = 1e-4
        self.policy_lr = 3e-4
        
        # PPO parameters
        self.ppo_epochs = 4
        self.ppo_clip_ratio = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.gae_lambda = 0.95
        self.discount_factor = 0.99
        
        # Training parameters
        self.max_episodes = 5000
        self.episodes_per_task = 4      # Episodes per task before sampling new task
        self.batch_size = 64            # Batch size for PPO updates
        self.vae_batch_size = 32        # Separate batch size for VAE training
        
        # Logging and saving
        self.log_interval = 10
        self.save_interval = 100
        self.eval_interval = 50
        self.eval_episodes = 10
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # VAE training
        self.vae_beta = 0.1            # KL divergence weight
        self.vae_update_freq = 1        # Update VAE every N episodes

def prepare_dataset(config):
    """Load and prepare dataset for meta-learning"""
    logger.info("Loading dataset...")
    
    # Load preprocessed data
    if not Path(config.data_path).exists():
        logger.info("Dataset not found, creating from scratch...")
        from environments.data_preparation import create_dataset
        config.data_path = create_dataset(config.data_path)
    
    # Create dataset wrapper
    dataset = Dataset(config.data_path)
    
    logger.info(f"Dataset loaded: {len(dataset)} days, {dataset.num_assets} assets, "
               f"{dataset.num_features} features")
    
    # Verify we have expected number of assets
    if dataset.num_assets != config.num_assets:
        logger.warning(f"Config expects {config.num_assets} assets, "
                      f"but dataset has {dataset.num_assets}. Updating config.")
        config.num_assets = dataset.num_assets
    
    # Create dataset tensor for MetaEnv
    # Get all data as a single tensor: (T, N, F)
    all_windows = []
    features_list = []
    prices_list = []
    
    for start_idx in range(0, len(dataset) - config.seq_len, config.seq_len):
        window = dataset.get_window(start_idx, start_idx + config.seq_len)
        all_windows.append(window)
        features_list.append(torch.tensor(window['features'], dtype=torch.float32))
        prices_list.append(torch.tensor(window['raw_prices'], dtype=torch.float32))
    
    # Stack all windows
    all_features = torch.stack(features_list)    # (num_windows, seq_len, N, F)
    all_prices = torch.stack(prices_list)        # (num_windows, seq_len, N)
    
    logger.info(f"Created {len(all_features)} task windows of length {config.seq_len}")
    
    # Reshape for MetaEnv: (total_time, N, F)
    dataset_tensor = {
        'features': all_features.view(-1, config.num_assets, dataset.num_features),
        'raw_prices': all_prices.view(-1, config.num_assets)
    }
    
    return dataset_tensor, dataset.feature_cols, dataset

def initialize_models(config, obs_shape, feature_columns):
    """Initialize VAE and Policy models"""
    logger.info("Initializing models...")
    
    device = torch.device(config.device)
    
    # Initialize VAE
    vae = VAE(
        obs_dim=obs_shape,
        num_assets=config.num_assets,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim
    ).to(device)
    
    # Initialize Policy
    policy = PortfolioPolicy(
        obs_shape=obs_shape,
        latent_dim=config.latent_dim,
        num_assets=config.num_assets,
        hidden_dim=config.hidden_dim
    ).to(device)
    
    logger.info(f"Models initialized on device: {device}")
    logger.info(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
    logger.info(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    return vae, policy

def evaluate_policy(env, policy, vae, config, num_episodes=10):
    """Evaluate policy performance"""
    logger.info(f"Evaluating policy over {num_episodes} episodes...")
    
    device = torch.device(config.device)
    total_rewards = []
    episode_lengths = []
    
    vae.eval()
    policy.eval()
    
    with torch.no_grad():
        for episode in range(num_episodes):
            # Sample and set new task
            task = env.sample_task()
            env.set_task(task)
            
            # Reset environment
            obs = env.reset()  # (N, F)
            obs_tensor = torch.ascontiguous_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # (1, N, F)
            
            episode_reward = 0
            episode_length = 0
            done = False
            
            # Initialize with zero latent for first step
            latent = torch.zeros(1, config.latent_dim, device=device)
            
            while not done:
                # Get action from policy
                action, value = policy.act(obs_tensor, latent, deterministic=True)
                action_cpu = action.squeeze(0).cpu().numpy()  # (N,)
                
                # Take environment step
                next_obs, reward, done, info = env.step(action_cpu)
                
                episode_reward += reward
                episode_length += 1
                
                # Update observation
                if not done:
                    obs_tensor = torch.ascontiguous_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                
                # TODO: Update latent using VAE encoder with trajectory history
                # For now, keep latent fixed during episode
                
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
    
    vae.train()
    policy.train()
    
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)
    
    logger.info(f"Evaluation complete: avg_reward={avg_reward:.4f}, avg_length={avg_length:.1f}")
    
    return {
        'avg_reward': avg_reward,
        'std_reward': np.std(total_rewards),
        'avg_length': avg_length,
        'rewards': total_rewards
    }

def main():
    """Main training loop"""
    # Setup logging
    exp_logger = setup_experiment_logging("varibad_portfolio")
    
    # Initialize configuration
    config = Config()
    logger.info("Starting VariBAD Portfolio Training")
    logger.info(f"Device: {config.device}")
    logger.info(f"Configuration: max_episodes={config.max_episodes}, "
               f"seq_len={config.seq_len}, latent_dim={config.latent_dim}")
    
    # Initialize variables early to avoid scope issues
    episodes_trained = 0
    best_reward = float('-inf')
    vae = None
    policy = None
    
    try:
        # Prepare dataset
        dataset_tensor, feature_columns, dataset_wrapper = prepare_dataset(config)
        
        # Initialize environment
        env = MetaEnv(
            dataset=dataset_tensor,
            feature_columns=feature_columns,
            seq_len=config.seq_len,
            min_horizon=config.min_horizon,
            max_horizon=config.max_horizon
        )
        
        # Get observation shape
        task = env.sample_task()
        env.set_task(task)
        initial_obs = env.reset()
        obs_shape = initial_obs.shape  # (N, F)
        
        logger.info(f"Environment initialized: obs_shape={obs_shape}")
        
        # Initialize models
        vae, policy = initialize_models(config, obs_shape, feature_columns)
        
        # Initialize trainer
        trainer = PPOTrainer(
            env=env,
            policy=policy,
            vae=vae,
            config=config,
            logger=exp_logger
        )
        
        # Training loop
        logger.info("Starting training...")
        
        while episodes_trained < config.max_episodes:
            # Sample new task
            task = env.sample_task()
            env.set_task(task)
            
            logger.info(f"New task sampled: task_id={task.get('task_id', 'unknown')}")
            
            # Train multiple episodes on this task
            task_rewards = []
            
            for episode_in_task in range(config.episodes_per_task):
                episode_result = trainer.train_episode()
                task_rewards.append(episode_result['episode_reward'])
                episodes_trained += 1
                
                # Logging
                if episodes_trained % config.log_interval == 0:
                    avg_task_reward = np.mean(task_rewards)
                    logger.info(f"Episode {episodes_trained:4d}: "
                               f"reward={episode_result['episode_reward']:8.4f}, "
                               f"task_avg={avg_task_reward:8.4f}, "
                               f"vae_loss={episode_result.get('vae_loss', 0):6.4f}, "
                               f"policy_loss={episode_result.get('policy_loss', 0):6.4f}")
                
                # Early stopping check
                if episodes_trained >= config.max_episodes:
                    break
            
            # Task-level logging
            avg_task_reward = np.mean(task_rewards)
            
            # Save best model
            if avg_task_reward > best_reward:
                best_reward = avg_task_reward
                save_path = Path(exp_logger.run_dir) / "best_model.pt"
                torch.save({
                    'episode': episodes_trained,
                    'vae_state_dict': vae.state_dict(),
                    'policy_state_dict': policy.state_dict(),
                    'best_reward': best_reward,
                    'config': config.__dict__
                }, save_path)
                
                logger.info(f"New best task reward: {best_reward:.4f} (saved to {save_path})")
            
            # Periodic evaluation
            if episodes_trained % config.eval_interval == 0:
                eval_results = evaluate_policy(env, policy, vae, config, config.eval_episodes)
                logger.info(f"Evaluation at episode {episodes_trained}: "
                           f"avg_reward={eval_results['avg_reward']:.4f} ± "
                           f"{eval_results['std_reward']:.4f}")
            
            # Periodic checkpointing
            if episodes_trained % config.save_interval == 0:
                checkpoint_path = Path(exp_logger.run_dir) / f"checkpoint_ep{episodes_trained}.pt"
                torch.save({
                    'episode': episodes_trained,
                    'vae_state_dict': vae.state_dict(),
                    'policy_state_dict': policy.state_dict(),
                    'trainer_state': trainer.get_state(),
                    'config': config.__dict__
                }, checkpoint_path)
                
                logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    
    finally:
        # Cleanup and final save
        logger.info("Training completed. Cleaning up...")
        
        if vae is not None and policy is not None:
            final_path = Path(exp_logger.run_dir) / "final_model.pt"
            torch.save({
                'episode': episodes_trained,
                'vae_state_dict': vae.state_dict(),
                'policy_state_dict': policy.state_dict(),
                'final_reward': best_reward,
                'config': config.__dict__
            }, final_path)
            
            logger.info(f"Final model saved: {final_path}")
        
        exp_logger.close()
        logger.info(f"Training finished. Total episodes: {episodes_trained}")

if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()