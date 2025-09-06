import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# Your existing imports
from environments.dataset import create_split_datasets
from environments.env import MetaEnv
from models.vae import VAE
from models.policy import PortfolioPolicy
from algorithms.trainer import PPOTrainer
from run_logger import seed_everything

logger = logging.getLogger(__name__)



@dataclass
class ValidationConfig:
    # Trial identification
    trial_id: int = 0
    seed: int = 42
    exp_name: str = "validation"
    
    # Hardcoded balanced config
    latent_dim: int = 512
    hidden_dim: int = 1024
    vae_lr: float = 0.0010748206602172
    policy_lr: float = 0.0020289998766945
    vae_beta: float = 0.0125762666385515
    vae_update_freq: int = 5
    seq_len: int = 200
    episodes_per_task: int = 3
    batch_size: int = 8192
    vae_batch_size: int = 1024
    ppo_epochs: int = 8
    entropy_coef: float = 0.0013141391952945
    num_envs: int = 200

    # Training schedule
    max_episodes: int = 6000
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.02
    val_interval: int = 200
    min_episodes_before_stopping: int = 1500
    val_episodes: int = 50
    
    # Environment
    data_path: str = "environments/data/sp500_rl_ready_cleaned.parquet"
    train_end: str = "2015-12-31"
    val_end: str = "2020-12-31"
    num_assets: int = 30
    device: str = "cuda"
    max_horizon: int = 160
    min_horizon: int = 145
    
    # Fixed PPO params
    ppo_clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    gae_lambda: float = 0.95
    discount_factor: float = 0.99

    # Ablation study flag
    disable_vae: bool = False

    # Debug flags
    debug_mode: bool = True
    debug_interval: int = 10  # Log every 10 episodes

def create_seed_configs(num_seeds: int = 25) -> List[ValidationConfig]:
    """Create 25 configs with different seeds plus 1 ablation config"""
    configs = []
    
    # Regular 25 seed runs
    for i in range(num_seeds):
        configs.append(ValidationConfig(seed=i, exp_name=f"validation_seed_{i}"))
    
    # Add one ablation run (VAE disabled)
    configs.append(ValidationConfig(
        seed=99,  # Use different seed for ablation
        exp_name="ablation_no_vae",
        disable_vae=True
    ))
    
    return configs

@dataclass
class RunResult:
    """Results from a single seed run"""
    seed: int
    episodes_trained: int
    best_val_sharpe: float
    final_val_sharpe: float
    early_stopped: bool
    training_time: float
    model_state: Optional[Dict] = None

class EarlyStoppingTracker:
    """Robust early stopping with minimum episode requirement"""
    
    def __init__(self, patience: int, min_delta: float, min_episodes: int):
        self.patience = patience
        self.min_delta = min_delta
        self.min_episodes = min_episodes
        self.best_score = float('-inf')
        self.patience_counter = 0
        self.stopped = False
        self.validation_scores = []
    
    def check(self, score: float, episode: int) -> bool:
        """Returns True if should stop early"""
        self.validation_scores.append(score)
        
        # Don't stop before minimum episodes
        if episode < self.min_episodes:
            return False
        
        # Check for improvement
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.patience_counter = 0
            logger.info(f"New best validation score: {self.best_score:.4f}")
            return False
        
        self.patience_counter += 1
        logger.info(f"No improvement. Patience: {self.patience_counter}/{self.patience}")
        
        if self.patience_counter >= self.patience:
            self.stopped = True
            logger.info(f"Early stopping triggered at episode {episode}")
            return True
        
        return False

class ExperimentRunner:
    """Orchestrates multi-seed validation experiments"""
    
    def __init__(self, results_dir: str = "validation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Will be set during setup
        self.split_tensors = None
        self.environments = None
        
    def setup_data_environment(self, config: ValidationConfig):
        """Setup datasets and environments (once for all seeds)"""
        logger.info("Setting up datasets and environments...")
        
        # Create splits
        if not Path(config.data_path).exists():
            logger.info("Dataset not found, creating from scratch.")
            from environments.data_preparation import create_dataset
            config.data_path = create_dataset(config.data_path)
        
        datasets = create_split_datasets(
            data_path=config.data_path,
            train_end=config.train_end,
            val_end=config.val_end
        )
        
        # Convert to tensors (same logic as your existing code)
        self.split_tensors = {}
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
            
            all_features = torch.stack(features_list)
            all_prices = torch.stack(prices_list)
            
            self.split_tensors[split_name] = {
                'features': all_features.view(-1, config.num_assets, dataset.num_features),
                'raw_prices': all_prices.view(-1, config.num_assets),
                'feature_columns': dataset.feature_cols,
                'num_windows': len(features_list)
            }
        
        # Create environments
        self.environments = {}
        for split_name, tensor_data in self.split_tensors.items():
            self.environments[split_name] = MetaEnv(
                dataset={
                    'features': tensor_data['features'],
                    'raw_prices': tensor_data['raw_prices']
                },
                feature_columns=tensor_data['feature_columns'],
                seq_len=config.seq_len,
                min_horizon=config.min_horizon,
                max_horizon=config.max_horizon
            )
        
        logger.info("Data environment setup complete")
    
    def run_single_seed(self, config: ValidationConfig) -> RunResult:
        start_time = datetime.now()
        logger.info(f"=== STARTING SEED {config.seed} DEBUG ===")
        logger.info(f"Config: num_envs={config.num_envs}, max_episodes={config.max_episodes}")
        
        # Set seed for reproducibility
        seed_everything(config.seed)
        
        # Initialize models
        train_env = self.environments['train']
        val_env = self.environments['val']
        
        # Get observation shape
        task = train_env.sample_task()
        train_env.set_task(task)
        initial_obs = train_env.reset()
        obs_shape = initial_obs.shape
        
        # Create models
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
        
        # Initialize trainer
        trainer = PPOTrainer(env=train_env, policy=policy, vae=vae, config=config)

        logger.info(f"DEBUG: Trainer configuration:")
        logger.info(f"  trainer.num_envs: {trainer.num_envs}")
        logger.info(f"  trainer.device: {trainer.device}")
        logger.info(f"  config.num_envs: {config.num_envs}")
        
        # Early stopping tracker
        early_stopping = EarlyStoppingTracker(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            min_episodes=config.min_episodes_before_stopping
        )
        
        # Training loop
        episodes_trained = 0
        best_val_sharpe = float('-inf')
        final_val_sharpe = None
        best_model_state = None
        
            # Add timing tracking
        episode_times = []
        last_log_time = start_time
        logger.info(f"Starting training for seed {config.seed}")
        
        while episodes_trained < config.max_episodes:
            # Sample new task
            task_start = datetime.now()
            task = train_env.sample_task()
            train_env.set_task(task)
            
            # Train episodes on this task
            for _ in tqdm(range(config.episodes_per_task), 
                         desc=f"Task episodes (Seed {config.seed})", 
                         leave=False, unit="episode"):

                episode_start = datetime.now()
            
                # DEBUG: Log trainer state
                if config.debug_mode and episodes_trained % config.debug_interval == 0:
                    logger.info(f"DEBUG Seed {config.seed}: Starting episode {episodes_trained}")
                    logger.info(f"  Trainer num_envs: {getattr(trainer, 'num_envs', 'NOT_SET')}")
                    logger.info(f"  GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")

                episode_result = trainer.train_episode()
                episode_end = datetime.now()
                episode_duration = (episode_end - episode_start).total_seconds()
                episode_times.append(episode_duration)
            
                episodes_trained += 1

                # DEBUG: Episode-level logging
                if config.debug_mode and episodes_trained % config.debug_interval == 0:
                    avg_episode_time = np.mean(episode_times[-config.debug_interval:])
                    logger.info(f"DEBUG Seed {config.seed}, Episode {episodes_trained}:")
                    logger.info(f"  Episode time: {episode_duration:.2f}s (avg last {config.debug_interval}: {avg_episode_time:.2f}s)")
                    logger.info(f"  Episode result keys: {episode_result.keys()}")
                    for key, value in episode_result.items():
                        if isinstance(value, (int, float)):
                            logger.info(f"    {key}: {value}")
                
                # Validation check
                if episodes_trained % config.val_interval == 0:
                    val_start = datetime.now()

                    logger.info(f"=== VALIDATION DEBUG Seed {config.seed}, Episode {episodes_trained} ===")

                    val_results = self.evaluate_on_split(
                        val_env, policy, vae, config, config.val_episodes
                    )

                    val_end = datetime.now()
                    val_duration = (val_end - val_start).total_seconds()

                    current_val_sharpe = val_results['avg_reward']
                    final_val_sharpe = current_val_sharpe
                    
                    # DEBUG: Detailed validation logging
                    logger.info(f"VALIDATION RESULTS:")
                    logger.info(f"  Time: {val_duration:.2f}s for {config.val_episodes} episodes")
                    logger.info(f"  Raw validation results: {val_results}")
                    logger.info(f"  Sharpe ratio: {current_val_sharpe:.6f}")
                    logger.info(f"  Reward stats: mean={val_results['avg_reward']:.6f}, std={val_results['std_reward']:.6f}")
                    logger.info(f"  Return stats: mean={val_results['avg_return']:.6f}, std={val_results['std_return']:.6f}")

                    logger.info(f"Seed {config.seed}, Episode {episodes_trained}: "
                              f"val_sharpe={current_val_sharpe:.4f}")
                    
                    # Track best model
                    if current_val_sharpe > best_val_sharpe:
                        best_val_sharpe = current_val_sharpe
                        best_model_state = {
                            'vae_state_dict': vae.state_dict(),
                            'policy_state_dict': policy.state_dict(),
                            'episodes_trained': episodes_trained,
                            'val_sharpe': current_val_sharpe
                        }
                    logger.info(f"  NEW BEST MODEL: {best_val_sharpe:.6f}")
                    
                    # Early stopping check
                    if early_stopping.check(current_val_sharpe, episodes_trained):
                        break
                
                if episodes_trained >= config.max_episodes:
                    break
            
            # DEBUG: Task-level timing
            task_end = datetime.now()
            task_duration = (task_end - task_start).total_seconds()
            
            if config.debug_mode:
                time_since_last = (task_end - last_log_time).total_seconds()
                logger.info(f"DEBUG Task completed: {config.episodes_per_task} episodes in {task_duration:.2f}s")
                logger.info(f"  Time since last log: {time_since_last:.2f}s")
                logger.info(f"  Episodes/minute: {(config.episodes_per_task * 60) / task_duration:.2f}")
                last_log_time = task_end

            if early_stopping.stopped or episodes_trained >= config.max_episodes:
                break
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        return RunResult(
            seed=config.seed,
            episodes_trained=episodes_trained,
            best_val_sharpe=best_val_sharpe,
            final_val_sharpe=final_val_sharpe or best_val_sharpe,
            early_stopped=early_stopping.stopped,
            training_time=training_time,
            model_state=best_model_state
        )
    
    def evaluate_on_split(self, split_env, policy, vae, config, num_episodes):
        """Evaluate policy on validation split with debug logging"""
        logger.info(f"DEBUG: Starting evaluation with {num_episodes} episodes")

        device = torch.device(config.device)
        episode_rewards = []
        episode_returns = []

        vae.eval()
        policy.eval()
        
        with torch.no_grad():
            # Debug first 5 episodes in detail
            for episode in range(min(5, num_episodes)):
                logger.info(f"DEBUG: Starting evaluation episode {episode}")

                task = split_env.sample_task()
                split_env.set_task(task)
                obs = split_env.reset()
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                
                episode_reward = 0
                step_rewards = []
                step_capitals = [split_env.initial_capital]
                done = False
                initial_capital = split_env.initial_capital
                trajectory_context = {'observations': [], 'actions': [], 'rewards': []}
                
                step_count = 0
                while not done and step_count < 10:  # Debug first 10 steps
                    # Get latent - handle ablation case
                    if config.disable_vae:
                        latent = torch.zeros(1, config.latent_dim, device=device)
                    elif len(trajectory_context['observations']) == 0:
                        latent = torch.zeros(1, config.latent_dim, device=device)
                    else:
                        obs_seq = torch.stack(trajectory_context['observations']).unsqueeze(0)
                        action_seq = torch.stack(trajectory_context['actions']).unsqueeze(0)
                        reward_seq = torch.stack(trajectory_context['rewards']).unsqueeze(0).unsqueeze(-1)
                        mu, logvar, _ = vae.encode(obs_seq, action_seq, reward_seq)
                        latent = vae.reparameterize(mu, logvar)
                    
                    # Get action
                    action, value = policy.act(obs_tensor, latent, deterministic=True)
                    action_cpu = action.squeeze(0).detach().cpu().numpy()

                    # DEBUG: Log action details
                    if episode < 2:
                        logger.info(f"  Step {step_count}: action_sum={action_cpu.sum():.6f}, action_max={action_cpu.max():.6f}")

                    next_obs, reward, done, info = split_env.step(action_cpu)
                    episode_reward += reward
                    step_rewards.append(reward)
                    step_capitals.append(split_env.current_capital)

                    # DEBUG: Log reward details
                    if episode < 2:
                        capital_change = split_env.current_capital - step_capitals[-2]
                        logger.info(f"    reward={reward:.6f}, capital_change={capital_change:.2f}")
                        logger.info(f"    current_capital={split_env.current_capital:.2f}")
                        
                    # Update context
                    trajectory_context['observations'].append(obs_tensor.squeeze(0).detach())
                    trajectory_context['actions'].append(action.squeeze(0).detach())
                    trajectory_context['rewards'].append(torch.tensor(reward, device=device))
                    
                    if not done:
                        obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    
                    step_count += 1
                
                final_capital = split_env.current_capital
                total_return = (final_capital - initial_capital) / initial_capital
                episode_rewards.append(episode_reward)
                episode_returns.append(total_return)

                # DEBUG: Episode summary
                if episode < 2:
                    logger.info(f"  Episode {episode} summary:")
                    logger.info(f"    Total reward: {episode_reward:.6f}")
                    logger.info(f"    Total return: {total_return:.6f}")
                    logger.info(f"    Steps: {len(step_rewards)}")
                    logger.info(f"    Avg step reward: {np.mean(step_rewards):.6f}")
                    logger.info(f"    Capital: {initial_capital:.2f} -> {final_capital:.2f}")

            # Run remaining episodes without detailed logging
            for episode in range(5, num_episodes):
                task = split_env.sample_task()
                split_env.set_task(task)
                obs = split_env.reset()
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                
                episode_reward = 0
                done = False
                initial_capital = split_env.initial_capital
                trajectory_context = {'observations': [], 'actions': [], 'rewards': []}
                
                while not done:
                    # Get latent - handle ablation case
                    if config.disable_vae:
                        latent = torch.zeros(1, config.latent_dim, device=device)
                    elif len(trajectory_context['observations']) == 0:
                        latent = torch.zeros(1, config.latent_dim, device=device)
                    else:
                        obs_seq = torch.stack(trajectory_context['observations']).unsqueeze(0)
                        action_seq = torch.stack(trajectory_context['actions']).unsqueeze(0)
                        reward_seq = torch.stack(trajectory_context['rewards']).unsqueeze(0).unsqueeze(-1)
                        mu, logvar, _ = vae.encode(obs_seq, action_seq, reward_seq)
                        latent = vae.reparameterize(mu, logvar)
                    
                    # Get action
                    action, value = policy.act(obs_tensor, latent, deterministic=True)
                    action_cpu = action.squeeze(0).detach().cpu().numpy()
                    next_obs, reward, done, info = split_env.step(action_cpu)
                    episode_reward += reward
                    
                    # Update context
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
        
        # Calculate Sharpe ratio with debugging
        rewards_array = np.array(episode_rewards)
        logger.info(f"DEBUG: Final evaluation stats:")
        logger.info(f"  Episode rewards: min={rewards_array.min():.6f}, max={rewards_array.max():.6f}, mean={rewards_array.mean():.6f}")
        logger.info(f"  Episode rewards std: {rewards_array.std():.6f}")
        logger.info(f"  Sharpe calculation: mean={rewards_array.mean():.6f} / std={rewards_array.std():.6f}")
        
        return {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns)
        }
    
    def run_all_seeds(self, configs: List[ValidationConfig]) -> List[RunResult]:
        """Run all seed configurations"""
        
        # Setup data environment once
        self.setup_data_environment(configs[0])  # All configs are same except seed
        
        results = []
        for i, config in enumerate(configs, 1):
            logger.info(f"Running configuration {i}/{len(configs)} (seed {config.seed})")
            
            try:
                result = self.run_single_seed(config)
                results.append(result)
                
                # Save intermediate results
                self.save_intermediate_result(result, i, len(configs))
                
                logger.info(f"Seed {config.seed} completed: "
                          f"best_val_sharpe={result.best_val_sharpe:.4f}, "
                          f"episodes={result.episodes_trained}, "
                          f"early_stopped={result.early_stopped}")
                
            except Exception as e:
                logger.error(f"Seed {config.seed} failed: {e}")
                continue
            
            # Cleanup GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(f"Completed {len(results)}/{len(configs)} seed runs")
        return results
    
    def save_intermediate_result(self, result: RunResult, current: int, total: int):
        """Save individual seed result"""
        result_file = self.results_dir / f"seed_{result.seed:02d}_result.json"
        
        result_data = {
            'seed': result.seed,
            'episodes_trained': result.episodes_trained,
            'best_val_sharpe': result.best_val_sharpe,
            'final_val_sharpe': result.final_val_sharpe,
            'early_stopped': result.early_stopped,
            'training_time': result.training_time,
            'progress': f"{current}/{total}"
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        # Save model checkpoint if it's the best so far
        if result.model_state:
            model_file = self.results_dir / f"seed_{result.seed:02d}_model.pt"
            torch.save(result.model_state, model_file)

class StatisticalAnalyzer:
    """Robust statistical analysis with IQM and bootstrap confidence intervals"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        
    def analyze_results(self, results: List[RunResult]) -> Dict[str, float]:
        """Compute robust statistics from all seed results"""
        if not results:
            raise ValueError("No results to analyze")
        
        # Extract validation scores
        val_scores = np.array([r.best_val_sharpe for r in results])
        
        # Core statistics
        stats_dict = {
            'num_runs': len(results),
            'mean': float(np.mean(val_scores)),
            'median': float(np.median(val_scores)),
            'std': float(np.std(val_scores)),
            'min': float(np.min(val_scores)),
            'max': float(np.max(val_scores)),
            'iqm': self.interquartile_mean(val_scores),
            'early_stop_rate': sum(r.early_stopped for r in results) / len(results)
        }
        
        # Bootstrap confidence intervals
        ci_low, ci_high = self.stratified_bootstrap(val_scores)
        stats_dict['iqm_ci_low'] = ci_low
        stats_dict['iqm_ci_high'] = ci_high
        
        # Episode statistics
        episodes = np.array([r.episodes_trained for r in results])
        stats_dict['avg_episodes'] = float(np.mean(episodes))
        stats_dict['median_episodes'] = float(np.median(episodes))
        
        logger.info(f"Statistical Analysis Results:")
        logger.info(f"  IQM: {stats_dict['iqm']:.4f} "
                   f"(95% CI: [{stats_dict['iqm_ci_low']:.4f}, {stats_dict['iqm_ci_high']:.4f}])")
        logger.info(f"  Mean±Std: {stats_dict['mean']:.4f} ± {stats_dict['std']:.4f}")
        logger.info(f"  Early stop rate: {stats_dict['early_stop_rate']:.2%}")
        
        return stats_dict
    
    def interquartile_mean(self, values: np.ndarray) -> float:
        """Compute IQM (mean of middle 50% of data)"""
        if len(values) == 0:
            return 0.0
        
        q25 = np.percentile(values, 25)
        q75 = np.percentile(values, 75)
        iqm_values = values[(values >= q25) & (values <= q75)]
        
        return float(np.mean(iqm_values))
    
    def stratified_bootstrap(self, scores: np.ndarray, n_resamples: int = 5000, 
                           confidence_level: float = 0.95) -> Tuple[float, float]:
        """Compute bootstrap confidence intervals for IQM"""
        bootstrap_iqms = []
        
        for _ in range(n_resamples):
            # Bootstrap resample
            resampled = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_iqms.append(self.interquartile_mean(resampled))
        
        bootstrap_iqms = np.array(bootstrap_iqms)
        
        # Confidence interval
        alpha = (1 - confidence_level) / 2
        ci_low = np.percentile(bootstrap_iqms, alpha * 100)
        ci_high = np.percentile(bootstrap_iqms, (1 - alpha) * 100)
        
        return float(ci_low), float(ci_high)
    
    def create_performance_profile(self, results: List[RunResult]) -> plt.Figure:
        """Create performance profile showing score distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        val_scores = [r.best_val_sharpe for r in results]
        episodes = [r.episodes_trained for r in results]
        early_stopped = [r.early_stopped for r in results]
        
        # Distribution of validation scores
        axes[0, 0].hist(val_scores, bins=15, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(np.mean(val_scores), color='red', linestyle='--', label='Mean')
        axes[0, 0].axvline(self.interquartile_mean(np.array(val_scores)), 
                          color='blue', linestyle='-', label='IQM')
        axes[0, 0].set_xlabel('Validation Sharpe Ratio')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Validation Scores')
        axes[0, 0].legend()
        
        # Box plot
        axes[0, 1].boxplot(val_scores)
        axes[0, 1].set_ylabel('Validation Sharpe Ratio')
        axes[0, 1].set_title('Score Distribution (Box Plot)')
        
        # Episodes vs Performance
        colors = ['red' if es else 'blue' for es in early_stopped]
        axes[1, 0].scatter(episodes, val_scores, c=colors, alpha=0.6)
        axes[1, 0].set_xlabel('Episodes Trained')
        axes[1, 0].set_ylabel('Validation Sharpe Ratio')
        axes[1, 0].set_title('Performance vs Training Length')
        
        # Early stopping analysis
        early_stop_counts = [sum(early_stopped), len(results) - sum(early_stopped)]
        labels = ['Early Stopped', 'Full Training']
        axes[1, 1].pie(early_stop_counts, labels=labels, autopct='%1.1f%%')
        axes[1, 1].set_title('Early Stopping Rate')
        
        plt.tight_layout()
        return fig
    
    def save_detailed_results(self, results: List[RunResult], stats: Dict[str, float]):
        """Save comprehensive results and analysis"""
        
        # Individual results CSV
        results_data = []
        for r in results:
            results_data.append({
                'seed': r.seed,
                'best_val_sharpe': r.best_val_sharpe,
                'final_val_sharpe': r.final_val_sharpe,
                'episodes_trained': r.episodes_trained,
                'early_stopped': r.early_stopped,
                'training_time': r.training_time
            })
        
        results_df = pd.DataFrame(results_data)
        results_csv = self.results_dir / "detailed_results.csv"
        results_df.to_csv(results_csv, index=False)
        
        # Summary statistics
        summary_file = self.results_dir / "statistical_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Performance profile plot
        fig = self.create_performance_profile(results)
        plot_file = self.results_dir / "performance_profile.png"
        fig.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Detailed results saved:")
        logger.info(f"  CSV: {results_csv}")
        logger.info(f"  Summary: {summary_file}")
        logger.info(f"  Plot: {plot_file}")


class BacktestEngine:
    """Handles model backtesting and benchmark comparison on test set"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.test_data = None
        self.obs_shape = None
    
    def setup_test_environment(self, config: ValidationConfig):
        """Setup test environment for backtesting"""
        logger.info("Setting up test environment for backtesting...")
        
        # Load test split
        datasets = create_split_datasets(
            data_path=config.data_path,
            train_end=config.train_end,
            val_end=config.val_end
        )
        test_dataset = datasets['test']
        
        # Extract full test window
        test_window = test_dataset.get_window(0, len(test_dataset))
        self.test_data = {
            'features': torch.tensor(test_window['features'], dtype=torch.float32),
            'raw_prices': torch.tensor(test_window['raw_prices'], dtype=torch.float32),
            'dates': self._generate_test_dates(test_window['features'].shape[0])
        }
        self.obs_shape = (test_window['features'].shape[1], test_window['features'].shape[2])
        
        logger.info(f"Test environment ready: {self.test_data['features'].shape[0]} days, "
                   f"{self.obs_shape[0]} assets, {self.obs_shape[1]} features")
    
    def _generate_test_dates(self, num_days: int) -> pd.DatetimeIndex:
        """Generate date index for test period"""
        # Start after validation end
        start_date = pd.to_datetime('2021-01-01')  # After val_end 2020-12-31
        return pd.date_range(start=start_date, periods=num_days, freq='D')
    
    def load_best_model(self, results: List[RunResult], config: ValidationConfig):
        """Load the best performing model from validation results"""
        if not results:
            raise ValueError("No results available to select best model")
        
        # Find best model by validation score
        best_result = max(results, key=lambda r: r.best_val_sharpe)
        
        if not best_result.model_state:
            raise ValueError("Best model has no saved state")
        
        logger.info(f"Loading best model: seed {best_result.seed}, "
                   f"val_sharpe={best_result.best_val_sharpe:.4f}")
        
        # Initialize models
        device = torch.device(config.device)
        vae = VAE(
            obs_dim=self.obs_shape,
            num_assets=config.num_assets,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim
        ).to(device)
        
        policy = PortfolioPolicy(
            obs_shape=self.obs_shape,
            latent_dim=config.latent_dim,
            num_assets=config.num_assets,
            hidden_dim=config.hidden_dim
        ).to(device)
        
        # Load states
        vae.load_state_dict(best_result.model_state['vae_state_dict'])
        policy.load_state_dict(best_result.model_state['policy_state_dict'])
        
        vae.eval()
        policy.eval()
        
        return vae, policy, best_result
    
    def backtest_model(self, vae, policy, config: ValidationConfig, model_name: str = "VariBAD") -> pd.DataFrame:
        """Backtest trained model on test set"""
        device = torch.device(config.device)
        features = self.test_data['features']
        prices = self.test_data['raw_prices']
        dates = self.test_data['dates']
        
        results = []
        current_wealth = 100000.0  # Initial capital
        trajectory_context = {'observations': [], 'actions': [], 'rewards': []}
        
        with torch.no_grad():
            for t in range(len(features) - 1):
                obs_t = features[t].unsqueeze(0).to(device)  # [1, N, F]
                
                # Get latent representation and handle ablation
                if config.disable_vae:
                    # Ablation: always use zero latent
                    latent = torch.zeros(1, config.latent_dim, device=device)
                if len(trajectory_context['observations']) == 0:
                    latent = torch.zeros(1, config.latent_dim, device=device)
                else:
                    obs_seq = torch.stack(trajectory_context['observations']).unsqueeze(0)
                    action_seq = torch.stack(trajectory_context['actions']).unsqueeze(0)
                    reward_seq = torch.stack(trajectory_context['rewards']).unsqueeze(0).unsqueeze(-1)
                    mu, logvar, _ = vae.encode(obs_seq, action_seq, reward_seq)
                    latent = vae.reparameterize(mu, logvar)
                
                # Get portfolio weights
                action, _ = policy.act(obs_t, latent, deterministic=True)
                weights = action.squeeze(0).detach().cpu().numpy()
                weights = np.clip(weights, 0, 1)
                weights = weights / weights.sum() if weights.sum() > 0 else weights
                
                # Calculate returns
                p_t = prices[t].cpu().numpy()
                p_t1 = prices[t + 1].cpu().numpy()
                asset_returns = (p_t1 - p_t) / (p_t + 1e-8)
                portfolio_return = np.dot(weights, asset_returns)
                
                # Update wealth
                current_wealth *= (1 + portfolio_return)
                
                # Store result
                results.append({
                    'date': dates[t + 1],
                    'wealth': current_wealth,
                    'returns': portfolio_return,
                    'model_name': model_name,
                    'strategy_type': 'varibad',
                    'kind': 'model'
                })
                
                # Update trajectory context
                trajectory_context['observations'].append(obs_t.squeeze(0))
                trajectory_context['actions'].append(action.squeeze(0))
                trajectory_context['rewards'].append(torch.tensor(portfolio_return, device=device))
                
                # Limit context length
                if len(trajectory_context['observations']) > 100:
                    for key in trajectory_context:
                        trajectory_context[key] = trajectory_context[key][-100:]
        
        return pd.DataFrame(results)


class BenchmarkStrategies:
   """Implementation of selected benchmark portfolio strategies"""
   
   @staticmethod
   def buy_and_hold(prices: np.ndarray, initial_weights: np.ndarray = None, **kwargs) -> np.ndarray:
       """Buy and hold strategy (rebalances to initial weights)"""
       if initial_weights is None:
           n_assets = len(prices)
           return np.ones(n_assets) / n_assets
       return initial_weights / initial_weights.sum()
   
   @staticmethod
   def random_allocation(prices: np.ndarray, **kwargs) -> np.ndarray:
       """Random portfolio allocation"""
       n_assets = len(prices)
       weights = np.random.rand(n_assets)
       return weights / weights.sum()
   
   @staticmethod
   def mean_variance_optimization(prices: np.ndarray, returns_history: List[np.ndarray], lookback: int = 60, **kwargs) -> np.ndarray:
       """
       Classical Markowitz mean-variance optimization.
       Uses past lookback returns to compute mean and covariance.
       Risk aversion parameter (lambda) controls risk vs return tradeoff.
       """
       if len(returns_history) < lookback:
           return BenchmarkStrategies.buy_and_hold(prices)
       
       recent_returns = np.array(returns_history[-lookback:])
       mu = np.mean(recent_returns, axis=0)          # expected returns
       cov = np.cov(recent_returns.T) + 1e-6 * np.eye(len(mu))  # covariance with small jitter
       
       # Simple risk-return tradeoff: maximize muᵀw - λ wᵀΣw
       # Here λ is a risk aversion parameter
       lambda_risk = 10.0
       inv_cov = np.linalg.pinv(cov)
       weights = inv_cov @ mu
       weights = np.clip(weights, 0, None)  # enforce long-only
       return weights / (weights.sum() + 1e-8)
   
   @staticmethod
   def bwsl(prices: np.ndarray, returns_history: List[np.ndarray], lookback: int = 60, **kwargs) -> np.ndarray:
       """
       Best/Worst Sharpe Long-only strategy:
       Picks the single asset with the highest Sharpe ratio over lookback window.
       """
       if len(returns_history) < lookback:
           return BenchmarkStrategies.buy_and_hold(prices)
       
       recent_returns = np.array(returns_history[-lookback:])
       mu = np.mean(recent_returns, axis=0)
       sigma = np.std(recent_returns, axis=0) + 1e-8
       sharpe_ratios = mu / sigma
       best_idx = np.argmax(sharpe_ratios)
       
       weights = np.zeros_like(mu)
       weights[best_idx] = 1.0
       return weights
   
   @staticmethod
   def momentum(prices: np.ndarray, returns_history: List[np.ndarray], lookback: int = 60, **kwargs) -> np.ndarray:
       """Momentum strategy based on past returns"""
       if len(returns_history) < lookback:
           return BenchmarkStrategies.buy_and_hold(prices)
       
       recent_returns = np.array(returns_history[-lookback:])
       momentum_scores = np.mean(recent_returns, axis=0)
       positive_momentum = np.clip(momentum_scores, 0, None)
       
       if positive_momentum.sum() == 0:
           return BenchmarkStrategies.buy_and_hold(prices)
       
       return positive_momentum / positive_momentum.sum()


def run_benchmark_backtests(test_data: Dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
   """Run all benchmark strategies on test data"""
   features = test_data['features']
   prices = test_data['raw_prices']
   
   benchmark_strategies = {
       'Buy_and_Hold': BenchmarkStrategies.buy_and_hold,
       'Random_Allocation': BenchmarkStrategies.random_allocation,
       'Mean_Variance_Optimization': BenchmarkStrategies.mean_variance_optimization,
       'Best_Worst_Sharpe_Long': BenchmarkStrategies.bwsl,
       'Momentum': BenchmarkStrategies.momentum
   }
   
   all_results = []
   
   for strategy_name, strategy_func in benchmark_strategies.items():
       logger.info(f"Running benchmark: {strategy_name}")
       
       current_wealth = 100000.0
       returns_history = []
       initial_weights = None
       
       # Set random seed for reproducible random allocation
       if strategy_name == 'Random_Allocation':
           np.random.seed(42)
       
       strategy_results = []
       
       for t in range(len(prices) - 1):
           p_t = prices[t].cpu().numpy()
           p_t1 = prices[t + 1].cpu().numpy()
           
           # Get strategy weights
           if strategy_name == 'Buy_and_Hold' and initial_weights is None:
               initial_weights = BenchmarkStrategies.buy_and_hold(p_t)
           
           weights = strategy_func(
               prices=p_t,
               returns_history=returns_history,
               initial_weights=initial_weights
           )
           
           # Calculate returns
           asset_returns = (p_t1 - p_t) / (p_t + 1e-8)
           portfolio_return = np.dot(weights, asset_returns)
           
           # Update wealth
           current_wealth *= (1 + portfolio_return)
           
           # Store result
           strategy_results.append({
               'date': dates[t + 1],
               'wealth': current_wealth,
               'returns': portfolio_return,
               'model_name': f'Benchmark_{strategy_name}',
               'strategy_type': strategy_name.lower(),
               'kind': 'benchmark'
           })
           
           # Update history
           returns_history.append(asset_returns)
           if len(returns_history) > 100:  # Limit history
               returns_history = returns_history[-100:]
       
       all_results.extend(strategy_results)
   
   return pd.DataFrame(all_results)


def run_unified_backtest(results: List[RunResult], config: ValidationConfig) -> pd.DataFrame:
   """Run unified backtest comparing best model vs benchmarks"""
   logger.info("Starting unified backtest analysis...")
   
   # Setup backtest engine
   backtest_engine = BacktestEngine(Path("validation_results"))
   backtest_engine.setup_test_environment(config)
   
   # Load and backtest best model
   vae, policy, best_result = backtest_engine.load_best_model(results, config)
   model_results = backtest_engine.backtest_model(vae, policy, config, "VariBAD_Best")
   
   # Run benchmark strategies
   benchmark_results = run_benchmark_backtests(
       backtest_engine.test_data, 
       backtest_engine.test_data['dates']
   )
   
   # Combine results
   combined_results = pd.concat([model_results, benchmark_results], ignore_index=True)
   
   # Save results
   results_file = Path("validation_results") / "backtest_comparison.csv"
   combined_results.to_csv(results_file, index=False)
   
   logger.info(f"Backtest comparison saved to: {results_file}")
   logger.info(f"Test period: {combined_results['date'].min()} to {combined_results['date'].max()}")
   
   return combined_results

def quick_performance_test():
    """Test different num_envs values to find optimal setting"""
    import time
    
    # Test configurations
    test_configs = [1, 4, 16, 50, 100, 200]
    
    for num_envs in test_configs:
        print(f"\n=== Testing num_envs = {num_envs} ===")
        
        # Create test config
        config = ValidationConfig(
            seed=42,
            num_envs=num_envs,
            max_episodes=20,  # Short test
            episodes_per_task=1,
            val_interval=999,  # No validation during test
            debug_mode=False   # No debug logging
        )
        
        try:
            # Setup (same as main)
            runner = ExperimentRunner("test_results")
            runner.setup_data_environment(config)
            
            # Quick training test
            start_time = time.time()
            result = runner.run_single_seed(config)
            end_time = time.time()
            
            duration = end_time - start_time
            episodes_per_second = config.max_episodes / duration
            
            print(f"  Duration: {duration:.2f}s")
            print(f"  Episodes/sec: {episodes_per_second:.2f}")
            print(f"  Episodes trained: {result.episodes_trained}")
            
            # Memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1024**3
                print(f"  Peak GPU memory: {memory_used:.2f} GB")
                torch.cuda.reset_peak_memory_stats()
        
        except Exception as e:
            print(f"  FAILED: {e}")
        
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Alternative: Profile the existing code
def profile_current_performance():
    """Profile current implementation to find bottlenecks"""
    import cProfile
    import pstats
    import io
    
    # Create a small test run
    config = ValidationConfig(
        seed=42,
        num_envs=200,  # Current setting
        max_episodes=10,  # Very short for profiling
        episodes_per_task=1,
        val_interval=999,
        debug_mode=False
    )
    
    pr = cProfile.Profile()
    pr.enable()
    
    try:
        runner = ExperimentRunner("profile_results")
        result = runner.run_single_seed(config)
    except Exception as e:
        print(f"Profiling failed: {e}")
    
    pr.disable()
    
    # Print top time consumers
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    print("=== PROFILING RESULTS ===")
    print(s.getvalue())

def main():
   """Complete validation pipeline with backtesting"""
   
   # Setup logging
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler('validation_pipeline.log'),
           logging.StreamHandler()
       ]
   )
   
   logger.info("="*60)
   logger.info("STARTING COMPLETE VALIDATION PIPELINE")
   logger.info("="*60)
   
   # Configuration
   num_seeds = 25
   configs = create_seed_configs(num_seeds)   
   logger.info(f"Configuration loaded: {len(configs)} total runs")
   logger.info(f"  - {num_seeds} regular seed runs")
   logger.info(f"  - 1 ablation run (VAE disabled)")
   results_dir = Path("validation_results")
   results_dir.mkdir(parents=True, exist_ok=True)
   
   try:
       # Phase 1: Multi-seed validation
       logger.info("PHASE 1: Multi-seed validation training")
       runner = ExperimentRunner(results_dir="validation_results")
       results = runner.run_all_seeds(configs)
       
       if len(results) < num_seeds * 0.8:
           logger.warning(f"Only {len(results)}/{num_seeds} runs completed successfully")
       
       # Phase 2: Statistical analysis
       logger.info("PHASE 2: Statistical analysis")
       analyzer = StatisticalAnalyzer(results_dir)
       stats = analyzer.analyze_results(results)
       analyzer.save_detailed_results(results, stats)
       
       # Phase 3: Backtesting
       logger.info("PHASE 3: Unified backtesting")
       combined_backtest = run_unified_backtest(results, configs[0])
       
       # Summary
       logger.info("="*60)
       logger.info("VALIDATION PIPELINE COMPLETE")
       logger.info("="*60)
       logger.info(f"Successfully completed: {len(results)}/{num_seeds} runs")
       logger.info(f"Final IQM Score: {stats['iqm']:.4f} "
                  f"(95% CI: [{stats['iqm_ci_low']:.4f}, {stats['iqm_ci_high']:.4f}])")
       logger.info(f"Results directory: {results_dir}")
       
       return stats, results, combined_backtest
       
   except Exception as e:
       logger.error(f"Pipeline failed: {e}")
       import traceback
       traceback.print_exc()
       raise


if __name__ == "__main__":
    import sys
    if "--performance-test" in sys.argv:
        quick_performance_test()
    elif "--profile" in sys.argv:
        profile_current_performance() 
    else:
        # Set GPU optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
        
        # Run validation
        stats, results, combined_backtest = main()

