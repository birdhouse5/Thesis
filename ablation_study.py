#!/usr/bin/env python3
"""
Ablation Study Implementation
Phase A: Core Component Ablations (VAE, Context Length, Task Adaptation, Meta-Learning)
"""

import torch
import logging
import numpy as np
from pathlib import Path
import json
import time
import argparse
from typing import Dict, Any

# Import your existing modules (same as main.py)
from environments.dataset import create_split_datasets
from environments.env import MetaEnv
from models.vae import VAE
from models.policy import PortfolioPolicy
from algorithms.trainer import PPOTrainer
from run_logger import RunLogger, seed_everything

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AblationConfig:
    """Ablation configuration based on optimal Phase 3 results"""
    def __init__(self, ablation_type: str, ablation_params: Dict[str, Any] = None):
        # Load optimal Phase 3 configuration as baseline
        self.load_optimal_config()
        
        # Apply ablation-specific modifications
        self.ablation_type = ablation_type
        self.apply_ablation(ablation_params or {})
        
        # Update experiment name
        self.exp_name = f"ablation_{ablation_type}_{int(time.time())}"
        
    def load_optimal_config(self):
        """Load the optimal configuration from Phase 3"""
        # Fixed optimal parameters from Phase 3
        self.latent_dim = 512
        self.hidden_dim = 1024
        self.vae_lr = 0.0010748206602172
        self.policy_lr = 0.0020289998766945
        self.vae_beta = 0.0125762666385515
        self.vae_update_freq = 5
        self.seq_len = 120
        self.episodes_per_task = 3
        self.batch_size = 8192
        self.vae_batch_size = 1024
        self.ppo_epochs = 8
        self.entropy_coef = 0.0013141391952945
        
        # Derived parameters
        self.max_horizon = min(self.seq_len - 10, int(self.seq_len * 0.8))
        self.min_horizon = max(self.max_horizon - 15, self.max_horizon // 2)
        
        # Base parameters
        self.data_path = "environments/data/sp500_rl_ready_cleaned.parquet"
        self.train_end = '2015-12-31'
        self.val_end = '2020-12-31'
        self.num_assets = 30
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Training parameters
        self.max_episodes = 3000  # Same as Phase 3
        self.val_interval = 500
        self.val_episodes = 50
        self.test_episodes = 100
        self.log_interval = 200
        self.save_interval = 1500
        
        # Fixed PPO parameters
        self.ppo_clip_ratio = 0.2
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.gae_lambda = 0.95
        self.discount_factor = 0.99
        
        # Seed for reproducibility
        self.seed = 42
        
        # Ablation flags (default: normal operation)
        self.disable_vae = False
        self.fixed_latent = False
        self.single_task = False
        
    def apply_ablation(self, params: Dict[str, Any]):
        """Apply specific ablation modifications"""
        if self.ablation_type == "no_vae":
            self.disable_vae = True
            logger.info("Ablation: VAE disabled - using zero latents")
            
        elif self.ablation_type == "context_short":
            self.seq_len = params.get("seq_len", 30)
            self.max_horizon = min(self.seq_len - 5, int(self.seq_len * 0.8))
            self.min_horizon = max(self.max_horizon - 10, self.max_horizon // 2)
            logger.info(f"Ablation: Short context - seq_len={self.seq_len}")
            
        elif self.ablation_type == "context_medium":
            self.seq_len = params.get("seq_len", 60)
            self.max_horizon = min(self.seq_len - 10, int(self.seq_len * 0.8))
            self.min_horizon = max(self.max_horizon - 15, self.max_horizon // 2)
            logger.info(f"Ablation: Medium context - seq_len={self.seq_len}")
            
        elif self.ablation_type == "context_long":
            self.seq_len = params.get("seq_len", 180)
            self.max_horizon = min(self.seq_len - 15, int(self.seq_len * 0.8))
            self.min_horizon = max(self.max_horizon - 20, self.max_horizon // 2)
            logger.info(f"Ablation: Long context - seq_len={self.seq_len}")
            
        elif self.ablation_type == "fixed_latent":
            self.fixed_latent = True
            logger.info("Ablation: Fixed random latent - no task adaptation")
            
        elif self.ablation_type == "single_task":
            self.single_task = True
            self.episodes_per_task = 20  # Much longer episodes on same task
            logger.info("Ablation: Single task training - no meta-learning")
            
        else:
            raise ValueError(f"Unknown ablation type: {self.ablation_type}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def run_ablation_experiment(ablation_type: str, ablation_params: Dict[str, Any] = None) -> Dict[str, float]:
    """Run a single ablation experiment"""
    logger.info(f"Starting ablation experiment: {ablation_type}")
    
    # Create ablation config
    config = AblationConfig(ablation_type, ablation_params)
    seed_everything(config.seed)
    
    # Set up run directory
    run_dir = Path("ablations") / "phase_A" / config.exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run = RunLogger(run_dir, config.to_dict(), name=config.exp_name)
    
    try:
        # Prepare datasets (same as main.py)
        split_tensors, split_datasets = prepare_split_datasets(config)
        
        # Create environments
        train_env = create_meta_env(
            split_tensors['train'], split_tensors['train']['feature_columns'], config
        )
        val_env = create_meta_env(
            split_tensors['val'], split_tensors['val']['feature_columns'], config
        )
        test_env = create_meta_env(
            split_tensors['test'], split_tensors['test']['feature_columns'], config
        )
        
        # Initialize models
        task = train_env.sample_task()
        train_env.set_task(task)
        initial_obs = train_env.reset()
        obs_shape = initial_obs.shape
        
        vae, policy = initialize_models(config, obs_shape, split_tensors['train']['feature_columns'])
        
        # Initialize trainer
        trainer = PPOTrainer(env=train_env, policy=policy, vae=vae, config=config)
        
        # Training loop (simplified from main.py)
        episodes_trained = 0
        best_val_sharpe = float('-inf')
        
        logger.info(f"Starting training for ablation: {ablation_type}")
        
        while episodes_trained < config.max_episodes:
            # Sample task
            if config.single_task:
                # Single task ablation: use same task for multiple episodes
                if episodes_trained % (config.episodes_per_task * 5) == 0:
                    task = train_env.sample_task()
                    train_env.set_task(task)
            else:
                # Normal: new task for each batch of episodes
                task = train_env.sample_task()
                train_env.set_task(task)
            
            # Train episodes on this task
            for _ in range(config.episodes_per_task):
                episode_result = trainer.train_episode()
                
                run.log_train_episode(
                    episodes_trained,
                    reward=episode_result.get('episode_reward'),
                    sharpe=episode_result.get('sharpe_ratio', episode_result.get('episode_reward')),
                    cum_wealth=episode_result.get('cumulative_return')
                )
                
                episodes_trained += 1
                
                # Validation
                if episodes_trained % config.val_interval == 0:
                    val_results = evaluate_on_split(
                        val_env, policy, vae, config, config.val_episodes, 'validation'
                    )
                    current_val_sharpe = val_results['avg_reward']
                    
                    run.log_val(
                        episodes_trained,
                        sharpe=current_val_sharpe,
                        reward=current_val_sharpe,
                        cum_wealth=val_results['avg_return']
                    )
                    
                    if current_val_sharpe > best_val_sharpe:
                        best_val_sharpe = current_val_sharpe
                        # Save best model
                        save_path = run_dir / "best_model.pt"
                        torch.save({
                            'ablation_type': ablation_type,
                            'episodes_trained': episodes_trained,
                            'vae_state_dict': vae.state_dict(),
                            'policy_state_dict': policy.state_dict(),
                            'best_val_sharpe': best_val_sharpe,
                            'config': config.to_dict()
                        }, save_path)
                    
                    logger.info(
                        f"Ablation {ablation_type}, Episode {episodes_trained}: "
                        f"val_sharpe={current_val_sharpe:.4f} (best={best_val_sharpe:.4f})"
                    )
                
                if episodes_trained >= config.max_episodes:
                    break
            
            if episodes_trained >= config.max_episodes:
                break
        
        # Final test evaluation
        logger.info(f"Final test evaluation for ablation: {ablation_type}")
        test_results = evaluate_on_split(test_env, policy, vae, config, config.test_episodes, 'test')
        
        run.log_test(
            sharpe=test_results['avg_reward'],
            reward=test_results['avg_reward'],
            cum_wealth=test_results['avg_return']
        )
        
        results = {
            'ablation_type': ablation_type,
            'best_val_sharpe': best_val_sharpe,
            'final_test_sharpe': test_results['avg_reward'],
            'episodes_trained': episodes_trained
        }
        
        logger.info(f"Ablation {ablation_type} completed:")
        logger.info(f"  Best val Sharpe: {best_val_sharpe:.4f}")
        logger.info(f"  Test Sharpe: {test_results['avg_reward']:.4f}")
        
        run.close()
        return results
        
    except Exception as e:
        logger.error(f"Ablation {ablation_type} failed: {str(e)}")
        run.close()
        raise


# Import helper functions from main.py (same logic)
def prepare_split_datasets(config):
    """Prepare train/val/test datasets (from main.py)"""
    if not Path(config.data_path).exists():
        logger.info("Dataset not found, creating from scratch.")
        from environments.data_preparation import create_dataset
        config.data_path = create_dataset(config.data_path)

    datasets = create_split_datasets(
        data_path=config.data_path,
        train_end=config.train_end,
        val_end=config.val_end
    )

    config.num_assets = datasets['train'].num_assets

    split_tensors = {}
    for split_name, dataset in datasets.items():
        features_list = []
        prices_list = []
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

        all_features = torch.stack(features_list)
        all_prices = torch.stack(prices_list)

        split_tensors[split_name] = {
            'features': all_features.view(-1, config.num_assets, dataset.num_features),
            'raw_prices': all_prices.view(-1, config.num_assets),
            'feature_columns': dataset.feature_cols,
            'num_windows': len(features_list)
        }

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
    device = torch.device(config.device)

    vae = VAE(
        obs_dim=obs_shape,
        num_assets=config.num_assets,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim
    ).to(device)

    policy = PortfolioPolicy(
        obs_shape=obs_shape,
        latent_dim=config.latent_dim,
        num_assets=config.num_assets,
        hidden_dim=config.hidden_dim
    ).to(device)

    return vae, policy


def evaluate_on_split(split_env, policy, vae, config, num_episodes, split_name):
    """Evaluate policy on a specific split (from main.py)"""
    device = torch.device(config.device)
    episode_rewards = []
    episode_returns = []

    vae.eval()
    policy.eval()

    with torch.no_grad():
        for episode in range(num_episodes):
            task = split_env.sample_task()
            split_env.set_task(task)
            obs = split_env.reset()
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            episode_reward = 0
            done = False
            initial_capital = split_env.initial_capital
            trajectory_context = {'observations': [], 'actions': [], 'rewards': []}

            while not done:
                # Handle ablation-specific latent generation
                if getattr(config, 'disable_vae', False):
                    latent = torch.zeros(1, config.latent_dim, device=device)
                elif getattr(config, 'fixed_latent', False):
                    # Fixed random latent (set seed for consistency)
                    torch.manual_seed(42)
                    latent = torch.randn(1, config.latent_dim, device=device)
                elif len(trajectory_context['observations']) == 0:
                    latent = torch.zeros(1, config.latent_dim, device=device)
                else:
                    obs_seq = torch.stack(trajectory_context['observations']).unsqueeze(0)
                    action_seq = torch.stack(trajectory_context['actions']).unsqueeze(0)
                    reward_seq = torch.stack(trajectory_context['rewards']).unsqueeze(0).unsqueeze(-1)
                    mu, logvar, _ = vae.encode(obs_seq, action_seq, reward_seq)
                    latent = vae.reparameterize(mu, logvar)

                action, value = policy.act(obs_tensor, latent, deterministic=True)
                action_cpu = action.squeeze(0).detach().cpu().numpy()
                next_obs, reward, done, info = split_env.step(action_cpu)
                episode_reward += reward

                trajectory_context['observations'].append(obs_tensor.squeeze(0).detach())
                trajectory_context['actions'].append(action.squeeze(0).detach())
                trajectory_context['rewards'].append(torch.tensor(reward, device=device))

                if not done:
                    obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

            final_capital = split_env.current_capital
            total_return = (final_capital - initial_capital) / initial_capital
            episode_rewards.append(episode_reward)
            episode_returns.append(total_return)

    vae.train()
    policy.train()

    return {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns)
    }


def main():
    """Run Phase A ablation studies"""
    parser = argparse.ArgumentParser(description="VariBAD Ablation Studies - Phase A")
    parser.add_argument("--ablations", type=str, default="no_vae", 
                       help="Comma-separated list of ablations to run")
    parser.add_argument("--parallel", action="store_true",
                       help="Run ablations in parallel (not implemented)")
    
    args = parser.parse_args()
    
    # Define available ablations
    available_ablations = {
        "no_vae": {},
        "context_short": {"seq_len": 30},
        "context_medium": {"seq_len": 60}, 
        "context_long": {"seq_len": 180},
        "fixed_latent": {},
        "single_task": {}
    }
    
    # Parse requested ablations
    requested_ablations = [a.strip() for a in args.ablations.split(",")]
    
    print("🧪 Starting Phase A Ablation Studies")
    print(f"Requested ablations: {requested_ablations}")
    print("=" * 60)
    
    results = {}
    
    for ablation_type in requested_ablations:
        if ablation_type not in available_ablations:
            logger.error(f"Unknown ablation: {ablation_type}")
            continue
            
        try:
            result = run_ablation_experiment(
                ablation_type, 
                available_ablations[ablation_type]
            )
            results[ablation_type] = result
            
        except Exception as e:
            logger.error(f"Failed to run ablation {ablation_type}: {str(e)}")
            continue
    
    # Save consolidated results
    results_dir = Path("ablations") / "phase_A"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "consolidated_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PHASE A ABLATION RESULTS SUMMARY")
    print("=" * 60)
    
    if results:
        print(f"{'Ablation':<15} {'Val Sharpe':<12} {'Test Sharpe':<12} {'Episodes':<10}")
        print("-" * 55)
        
        for ablation_type, result in results.items():
            print(f"{ablation_type:<15} "
                  f"{result['best_val_sharpe']:<12.4f} "
                  f"{result['final_test_sharpe']:<12.4f} "
                  f"{result['episodes_trained']:<10}")
    
    print(f"\nDetailed results saved to: {results_dir}")
    print("Next: Analyze results and compare to Phase 3 baseline")


if __name__ == "__main__":
    main()