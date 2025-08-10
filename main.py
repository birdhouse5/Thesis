import torch
import logging
import numpy as np
from pathlib import Path
import json
import argparse

from logger_config import setup_experiment_logging
from environments.data_preparation import load_dataset
from environments.dataset import Dataset, create_split_datasets
from environments.env import MetaEnv
from models.vae import VAE
from models.policy import PortfolioPolicy
from algorithms.trainer import PPOTrainer
from csv_logger import CSVLogger

logger = logging.getLogger(__name__)

class Config:
    """Training configuration that can load from JSON files"""
    def __init__(self, config_file=None):
        # Default values (your existing config)
        self.data_path = "environments/data/sp500_rl_ready_cleaned.parquet"
        self.train_end = '2015-12-31'
        self.val_end = '2020-12-31'
        
        # Environment parameters
        self.num_assets = 30
        self.seq_len = 60
        self.min_horizon = 45
        self.max_horizon = 60
        
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
        self.max_episodes = 10
        self.episodes_per_task = 2
        self.batch_size = 64
        self.vae_batch_size = 32
        
        # Validation and testing
        self.val_interval = 200
        self.val_episodes = 20
        self.test_episodes = 50
        
        # Logging
        self.log_interval = 10
        self.save_interval = 500
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # VAE training
        self.vae_beta = 0.1
        self.vae_update_freq = 1
        
        # Experiment name (will be set by config file)
        self.exp_name = "varibad_default"
        
        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file):
        """Load configuration from JSON file"""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Update attributes from config file
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config parameter: {key}")
        
        # Set experiment name from config file name if not specified
        if 'exp_name' not in config_dict:
            self.exp_name = config_path.stem
    
    def save_to_file(self, filepath):
        """Save current configuration to JSON file"""
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_') and not callable(v)}
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def __str__(self):
        return f"Config(exp_name={self.exp_name}, latent_dim={self.latent_dim}, hidden_dim={self.hidden_dim})"

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="VariBAD Portfolio Training")
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--exp_name", type=str, help="Override experiment name")
    return parser.parse_args()

def prepare_split_datasets(config):
    """Load and prepare datasets for train/val/test splits"""
    logger.info("Preparing datasets with temporal splits...")
    
    # Ensure dataset exists
    if not Path(config.data_path).exists():
        logger.info("Dataset not found, creating from scratch...")
        from environments.data_preparation import create_dataset
        config.data_path = create_dataset(config.data_path)
    
    # Create split datasets
    datasets = create_split_datasets(
        data_path=config.data_path,
        train_end=config.train_end,
        val_end=config.val_end
    )
    
    # Verify consistent number of assets across splits
    train_assets = datasets['train'].num_assets
    val_assets = datasets['val'].num_assets
    test_assets = datasets['test'].num_assets
    
    if not (train_assets == val_assets == test_assets):
        logger.warning(f"Inconsistent asset counts: train={train_assets}, val={val_assets}, test={test_assets}")
    
    # Update config with actual asset count
    config.num_assets = train_assets
    
    # Create dataset tensors for each split
    split_tensors = {}
    
    for split_name, dataset in datasets.items():
        logger.info(f"Creating dataset tensor for {split_name} split...")
        
        # Create windows for this split
        features_list = []
        prices_list = []
        
        # Calculate number of complete windows we can create
        num_windows = max(1, (len(dataset) - config.seq_len) // config.seq_len)
        
        for i in range(num_windows):
            start_idx = i * config.seq_len
            end_idx = start_idx + config.seq_len
            
            if end_idx <= len(dataset):
                window = dataset.get_window(start_idx, end_idx)
                features_list.append(torch.tensor(window['features'], dtype=torch.float32))
                prices_list.append(torch.tensor(window['raw_prices'], dtype=torch.float32))
        
        if len(features_list) == 0:
            raise ValueError(f"No complete windows available for {split_name} split")
        
        # Stack windows and reshape for MetaEnv
        all_features = torch.stack(features_list)    # (num_windows, seq_len, N, F)
        all_prices = torch.stack(prices_list)        # (num_windows, seq_len, N)
        
        # Reshape: (total_time, N, F)
        split_tensors[split_name] = {
            'features': all_features.view(-1, config.num_assets, dataset.num_features),
            'raw_prices': all_prices.view(-1, config.num_assets),
            'feature_columns': dataset.feature_cols,
            'num_windows': len(features_list)
        }
        
        logger.info(f"{split_name} split: {len(features_list)} windows, "
                   f"tensor shape {split_tensors[split_name]['features'].shape}")
    
    return split_tensors, datasets

def create_meta_env(dataset_tensor, feature_columns, config):
    """Create MetaEnv from dataset tensor"""
    return MetaEnv(
        dataset={
            'features': dataset_tensor['features'],
            'raw_prices': dataset_tensor['raw_prices']
        },
        feature_columns=feature_columns,
        seq_len=config.seq_len,
        min_horizon=config.min_horizon,
        max_horizon=config.max_horizon
    )

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

def evaluate_on_split(split_env, policy, vae, config, num_episodes, split_name):
    """Evaluate policy on a specific split"""
    logger.info(f"Evaluating on {split_name} split ({num_episodes} episodes)...")
    
    device = torch.device(config.device)
    episode_rewards = []
    episode_lengths = []
    episode_returns = []
    
    vae.eval()
    policy.eval()
    
    with torch.no_grad():
        for episode in range(num_episodes):
            # Sample and set new task
            task = split_env.sample_task()
            split_env.set_task(task)
            
            # Reset environment
            obs = split_env.reset()
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            episode_reward = 0
            episode_length = 0
            done = False
            initial_capital = split_env.initial_capital
            
            # Initialize trajectory context for VAE
            trajectory_context = {'observations': [], 'actions': [], 'rewards': []}
            
            while not done:
                # Get latent from VAE encoder
                if len(trajectory_context['observations']) == 0:
                    latent = torch.zeros(1, config.latent_dim, device=device)
                else:
                    obs_seq = torch.stack(trajectory_context['observations']).unsqueeze(0)
                    action_seq = torch.stack(trajectory_context['actions']).unsqueeze(0)
                    reward_seq = torch.stack(trajectory_context['rewards']).unsqueeze(0).unsqueeze(-1)
                    
                    mu, logvar, _ = vae.encode(obs_seq, action_seq, reward_seq)
                    latent = vae.reparameterize(mu, logvar)
                
                # Get action from policy
                action, value = policy.act(obs_tensor, latent, deterministic=True)
                
                # FIX: Ensure proper tensor conversion for environment
                action_cpu = action.squeeze(0).detach().cpu().numpy()
                
                # Take environment step
                next_obs, reward, done, info = split_env.step(action_cpu)
                
                episode_reward += reward
                episode_length += 1
                
                # Update trajectory context - keep tensors on device
                trajectory_context['observations'].append(obs_tensor.squeeze(0).detach())
                trajectory_context['actions'].append(action.squeeze(0).detach())
                trajectory_context['rewards'].append(torch.tensor(reward, device=device))
                
                # Update observation
                if not done:
                    obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Calculate total return
            final_capital = split_env.current_capital
            total_return = (final_capital - initial_capital) / initial_capital
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_returns.append(total_return)
    
    vae.train()
    policy.train()
    
    results = {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_length': np.mean(episode_lengths),
        'avg_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'rewards': episode_rewards,
        'returns': episode_returns
    }
    
    logger.info(f"{split_name} evaluation: "
               f"avg_reward={results['avg_reward']:.4f}±{results['std_reward']:.4f}, "
               f"avg_return={results['avg_return']:.4f}±{results['std_return']:.4f}")
    
    return results

def main():
    """Main training loop with train-test-val split"""

    args = parse_args()
    
    # Initialize configuration
    config = Config(args.config)
    if args.exp_name:
        config.exp_name = args.exp_name

    # Setup logging
    exp_logger = setup_experiment_logging("logs")
    csv_logger = CSVLogger(exp_logger.run_dir)

    csv_logger.log_config(config)
    logger.info("Starting VariBAD Portfolio Training with Train-Test-Val Split")
    logger.info(f"Device: {config.device}")
    logger.info(f"Split dates: train≤{config.train_end}, val={config.train_end} to {config.val_end}, test>{config.val_end}")
    
    # Initialize variables
    episodes_trained = 0
    best_val_reward = float('-inf')
    vae = None
    policy = None
    
    try:
        # Prepare split datasets
        split_tensors, split_datasets = prepare_split_datasets(config)
        
        # Create environments for each split
        train_env = create_meta_env(split_tensors['train'], split_tensors['train']['feature_columns'], config)
        val_env = create_meta_env(split_tensors['val'], split_tensors['val']['feature_columns'], config)
        test_env = create_meta_env(split_tensors['test'], split_tensors['test']['feature_columns'], config)
        
        # Get observation shape from training environment
        task = train_env.sample_task()
        train_env.set_task(task)
        initial_obs = train_env.reset()
        obs_shape = initial_obs.shape
        
        logger.info(f"Environment initialized: obs_shape={obs_shape}")
        
        # Initialize models
        vae, policy = initialize_models(config, obs_shape, split_tensors['train']['feature_columns'])
        
        # Initialize trainer (uses training environment)
        trainer = PPOTrainer(
            env=train_env,
            policy=policy,
            vae=vae,
            config=config,
            logger=exp_logger,
            csv_logger = csv_logger
        )

        # Training loop
        logger.info("Starting training on training split...")
        
        while episodes_trained < config.max_episodes:
            # Sample new task from training set
            task = train_env.sample_task()
            train_env.set_task(task)
            
            # Train multiple episodes on this task
            task_rewards = []
            
            for episode_in_task in range(config.episodes_per_task):
                episode_result = trainer.train_episode()
                
                csv_logger.log_training_step(episodes_trained, episode_result)
                csv_logger.log_episode_details(episodes_trained, {
                    **episode_result,
                    'task_id': getattr(train_env, 'task_id', None),
                    'cumulative_return': (train_env.current_capital - train_env.initial_capital) / train_env.initial_capital
                })

                task_rewards.append(episode_result['episode_reward'])
                episodes_trained += 1
                
                # Logging
                if episodes_trained % config.log_interval == 0:
                    val_results = evaluate_on_split(val_env, policy, vae, config, config.val_episodes, 'validation')
                    csv_logger.log_validation(episodes_trained, val_results)
                    avg_task_reward = np.mean(task_rewards)
                    logger.info(f"Episode {episodes_trained:4d}: "
                               f"reward={episode_result['episode_reward']:8.4f}, "
                               f"task_avg={avg_task_reward:8.4f}")
                
                # Early stopping check
                if episodes_trained >= config.max_episodes:
                    break
            
            # Validation
            if episodes_trained % config.val_interval == 0:
                val_results = evaluate_on_split(val_env, policy, vae, config, config.val_episodes, 'validation')
                
                # Log validation results
                exp_logger.log_scalar('val/avg_reward', val_results['avg_reward'], episodes_trained)
                exp_logger.log_scalar('val/avg_return', val_results['avg_return'], episodes_trained)
                
                # Save best model based on validation performance
                if val_results['avg_reward'] > best_val_reward:
                    best_val_reward = val_results['avg_reward']
                    save_path = Path(exp_logger.run_dir) / "best_model.pt"
                    torch.save({
                        'episode': episodes_trained,
                        'vae_state_dict': vae.state_dict(),
                        'policy_state_dict': policy.state_dict(),
                        'best_val_reward': best_val_reward,
                        'val_results': val_results,
                        'config': config.__dict__
                    }, save_path)
                    
                    logger.info(f"New best validation reward: {best_val_reward:.4f} (saved to {save_path})")
            
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
        # Final evaluation on test set
        if vae is not None and policy is not None:
            logger.info("=" * 60)
            logger.info("FINAL EVALUATION ON TEST SET")
            logger.info("=" * 60)
            
            # Load best model for testing
            best_model_path = Path(exp_logger.run_dir) / "best_model.pt"
            if best_model_path.exists():
                logger.info("Loading best model for final test...")
                checkpoint = torch.load(best_model_path, weights_only=False)
                vae.load_state_dict(checkpoint['vae_state_dict'])
                policy.load_state_dict(checkpoint['policy_state_dict'])
            
            # Test evaluation
            test_results = evaluate_on_split(test_env, policy, vae, config, config.test_episodes, 'test')
            
            # Log final test results

            csv_logger.log_test(test_results)
            csv_logger.save_all_csvs()
            
            print(f"CSV logs saved to: {csv_logger.experiment_dir}")
            print(f"  - Training: {csv_logger.training_csv}")
            print(f"  - Validation: {csv_logger.validation_csv}")
            print(f"  - Test: {csv_logger.test_csv}")
            print(f"  - Episodes: {csv_logger.episodes_csv}")
            print(f"  - Summary: {csv_logger.summary_csv}")

            exp_logger.log_scalar('test/final_avg_reward', test_results['avg_reward'], episodes_trained)
            exp_logger.log_scalar('test/final_avg_return', test_results['avg_return'], episodes_trained)
            
            # Save final results
            final_path = Path(exp_logger.run_dir) / "final_results.pt"
            torch.save({
                'episode': episodes_trained,
                'vae_state_dict': vae.state_dict(),
                'policy_state_dict': policy.state_dict(),
                'test_results': test_results,
                'config': config.__dict__
            }, final_path)
            
            logger.info(f"Final test results saved: {final_path}")
            logger.info(f"Training completed. Episodes: {episodes_trained}")
            logger.info(f"Final test performance: {test_results['avg_reward']:.4f}±{test_results['std_reward']:.4f}")
            
        exp_logger.close()

if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()