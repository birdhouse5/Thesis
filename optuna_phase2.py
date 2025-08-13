import torch
torch.set_num_threads(1)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    
import optuna
import logging
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, Any
import os

# Import your existing modules
from environments.dataset import create_split_datasets
from environments.env import MetaEnv
from models.vae import VAE
from models.policy import PortfolioPolicy
from algorithms.trainer import PPOTrainer
from run_logger import RunLogger, seed_everything

# Optuna-specific settings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptunaPhase2Config:
    """Phase 2 Configuration - VAE and Training Dynamics Optimization"""
    def __init__(self, trial: optuna.Trial):
        # FIXED from Phase 1 results (winning architecture)
        self.latent_dim = 512
        self.hidden_dim = 1024
        
        # REFINED learning rates (focus on successful high-LR region from best trial)
        self.vae_lr = trial.suggest_float('vae_lr', 1e-3, 5e-3, log=True)
        self.policy_lr = trial.suggest_float('policy_lr', 5e-4, 3e-3, log=True)
        
        # NEW Phase 2 parameters: VAE behavior
        self.vae_beta = trial.suggest_float('vae_beta', 0.001, 1.0, log=True)
        self.vae_update_freq = trial.suggest_categorical('vae_update_freq', [1, 2, 5])
        
        # NEW Phase 2 parameters: Context and sequence dynamics
        self.seq_len = trial.suggest_categorical('seq_len', [30, 60, 90, 120])
        self.episodes_per_task = trial.suggest_categorical('episodes_per_task', [3, 5, 8, 12])
        
        # NEW Phase 2 parameters: Batch processing
        self.batch_size = trial.suggest_categorical('batch_size', [1024, 2048, 4096, 8192])
        self.vae_batch_size = trial.suggest_categorical('vae_batch_size', [256, 512, 1024, 2048])
        
        # NEW Phase 2 parameters: PPO dynamics
        self.ppo_epochs = trial.suggest_categorical('ppo_epochs', [2, 4, 8])
        self.entropy_coef = trial.suggest_float('entropy_coef', 0.001, 0.1, log=True)
        
        # Ensure constraints
        self.vae_batch_size = min(self.vae_batch_size, self.batch_size // 2)
        
        # Adjust horizons based on seq_len
        self.max_horizon = min(self.seq_len - 10, int(self.seq_len * 0.8))
        self.min_horizon = max(self.max_horizon - 15, self.max_horizon // 2)
        
        # Fixed base parameters (successful from Phase 1)
        self.data_path = "environments/data/sp500_rl_ready_cleaned.parquet"
        self.train_end = '2015-12-31'
        self.val_end = '2020-12-31'
        self.num_assets = 30
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Extended training for Phase 2 (need more episodes to see VAE effects)
        self.max_episodes = 3000  # Increased from 2000
        self.val_interval = 600   # Check validation less frequently
        self.val_episodes = 50
        self.test_episodes = 100
        self.log_interval = 150
        self.save_interval = 1500
        
        # Fixed PPO parameters (keep successful defaults)
        self.ppo_clip_ratio = 0.2
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.gae_lambda = 0.95
        self.discount_factor = 0.99
        
        # Experiment naming
        self.exp_name = f"phase2_t{trial.number}_beta{self.vae_beta:.3f}_seq{self.seq_len}_ept{self.episodes_per_task}"
        
        # Seed for reproducibility
        self.seed = 42
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def update_vae_fixed(trainer):
    """Fixed VAE update with proper gradient handling (from Phase 1)"""
    if len(trainer.vae_buffer) < trainer.config.vae_batch_size:
        return 0.0

    indices = np.random.choice(len(trainer.vae_buffer), trainer.config.vae_batch_size, replace=False)
    batch_traj = [trainer.vae_buffer[i] for i in indices]

    total_loss = None
    loss_count = 0

    trainer.vae_optimizer.zero_grad()

    for tr in batch_traj:
        seq_len = len(tr["rewards"])
        if seq_len < 2:
            continue

        max_t = min(seq_len - 1, 20)  
        t = np.random.randint(1, max_t + 1)

        obs_ctx = tr["observations"][:t].unsqueeze(0)        
        act_ctx = tr["actions"][:t].unsqueeze(0)             
        rew_ctx = tr["rewards"][:t].unsqueeze(0).unsqueeze(-1)  

        vae_loss, _ = trainer.vae.compute_loss(
            obs_ctx, act_ctx, rew_ctx, beta=trainer.config.vae_beta, context_len=t
        )
        
        if total_loss is None:
            total_loss = vae_loss
        else:
            total_loss = total_loss + vae_loss
        loss_count += 1

    if loss_count == 0:
        return 0.0

    avg_loss = total_loss / loss_count
    avg_loss.backward()
    torch.nn.utils.clip_grad_norm_(trainer.vae.parameters(), trainer.config.max_grad_norm)
    trainer.vae_optimizer.step()
    
    return float(avg_loss.item())


def objective_phase2(trial: optuna.Trial) -> float:
    """
    Phase 2 objective function focusing on VAE and training dynamics.
    Returns validation Sharpe ratio to maximize.
    """
    import gc, time, torch
    try:
        # Create Phase 2 configuration
        config = OptunaPhase2Config(trial)
        
        # Per-trial seed offset
        seed_everything(config.seed + int(trial.number))

        # Set up run directory
        run_dir = Path("optuna_phase2_runs") / f"trial_{trial.number}_{int(time.time())}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run = RunLogger(run_dir, config.to_dict(), name=config.exp_name)

        logger.info(f"Phase 2 Trial {trial.number}: {config.exp_name}")
        logger.info(
            f"VAE params: beta={config.vae_beta:.4f}, seq_len={config.seq_len}, "
            f"update_freq={config.vae_update_freq}"
        )
        logger.info(
            f"Training params: batch_size={config.batch_size}, episodes_per_task={config.episodes_per_task}, "
            f"ppo_epochs={config.ppo_epochs}"
        )

        try:
            # Prepare datasets (same as Phase 1)
            split_tensors, split_datasets = prepare_split_datasets(config)

            # Create environments
            train_env = create_meta_env(
                split_tensors['train'], split_tensors['train']['feature_columns'], config
            )
            val_env = create_meta_env(
                split_tensors['val'], split_tensors['val']['feature_columns'], config
            )

            try:
                # Get observation shape
                task = train_env.sample_task()
                train_env.set_task(task)
                initial_obs = train_env.reset()
                obs_shape = initial_obs.shape

                # Initialize models with FIXED architecture
                vae, policy = initialize_models(
                    config, obs_shape, split_tensors['train']['feature_columns']
                )

                # Initialize trainer
                trainer = PPOTrainer(env=train_env, policy=policy, vae=vae, config=config)
                trainer.update_vae = lambda: update_vae_fixed(trainer)

                # Training loop with enhanced early stopping for Phase 2
                episodes_trained = 0
                best_val_sharpe = float('-inf')
                patience = 4  # Slightly more patience for longer training
                no_improvement_count = 0
                validation_history = []

                logger.info(f"Starting Phase 2 training for trial {trial.number}")

                while episodes_trained < config.max_episodes:
                    task = train_env.sample_task()
                    train_env.set_task(task)

                    for _ in range(config.episodes_per_task):
                        episode_result = trainer.train_episode()

                        run.log_train_episode(
                            episodes_trained,
                            reward=episode_result.get('episode_reward'),
                            sharpe=episode_result.get('sharpe_ratio', episode_result.get('episode_reward')),
                            cum_wealth=episode_result.get('cumulative_return'),
                        )

                        episodes_trained += 1

                        # Validation check
                        if episodes_trained % config.val_interval == 0:
                            val_results = evaluate_on_split(
                                val_env, policy, vae, config, config.val_episodes, 'validation'
                            )
                            current_val_sharpe = val_results['avg_reward']
                            validation_history.append(current_val_sharpe)

                            run.log_val(
                                episodes_trained,
                                sharpe=current_val_sharpe,
                                reward=current_val_sharpe,
                                cum_wealth=val_results['avg_return'],
                            )

                            logger.info(
                                f"Trial {trial.number}, Episode {episodes_trained}: "
                                f"val_sharpe={current_val_sharpe:.4f} "
                                f"(best={best_val_sharpe:.4f})"
                            )

                            # Enhanced early stopping logic
                            if current_val_sharpe > best_val_sharpe:
                                best_val_sharpe = current_val_sharpe
                                no_improvement_count = 0
                                save_path = run_dir / "best_model.pt"
                                torch.save(
                                    {
                                        'trial_number': trial.number,
                                        'episodes_trained': episodes_trained,
                                        'vae_state_dict': vae.state_dict(),
                                        'policy_state_dict': policy.state_dict(),
                                        'best_val_sharpe': best_val_sharpe,
                                        'config': config.to_dict(),
                                        'validation_history': validation_history,
                                    },
                                    save_path,
                                )
                            else:
                                no_improvement_count += 1

                            # Report to Optuna
                            trial.report(current_val_sharpe, episodes_trained)
                            
                            # Pruning with more context for Phase 2
                            if trial.should_prune():
                                logger.info(f"Trial {trial.number} pruned at episode {episodes_trained}")
                                try: run.close()
                                except Exception: pass
                                raise optuna.TrialPruned()

                            # Early stopping if clearly not improving
                            if no_improvement_count >= patience:
                                logger.info(f"Trial {trial.number} early stopped (no improvement for {patience} checks)")
                                break

                        if episodes_trained >= config.max_episodes:
                            break

                    if episodes_trained >= config.max_episodes:
                        break

                logger.info(f"Trial {trial.number} completed: best_val_sharpe={best_val_sharpe:.4f}")
                run.close()
                return best_val_sharpe

            except torch.cuda.OutOfMemoryError as e:
                logger.warning(
                    f"CUDA OOM on trial {trial.number}; pruning. "
                    f"(batch_size={getattr(config,'batch_size',None)}, "
                    f"seq_len={getattr(config,'seq_len',None)})"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try: run.close()
                except Exception: pass
                raise optuna.TrialPruned() from e

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"RuntimeError OOM on trial {trial.number}; pruning.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    try: run.close()
                    except Exception: pass
                    raise optuna.TrialPruned() from e
                raise

        finally:
            # Cleanup
            for name in ("trainer", "vae", "policy", "train_env", "val_env",
                         "split_tensors", "split_datasets", "initial_obs"):
                if name in locals():
                    del locals()[name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    except optuna.TrialPruned:
        raise

    except Exception as e:
        import traceback
        logger.error(f"Trial {trial.number} failed: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise

    finally:
        try:
            if 'run' in locals():
                run.close()
        except Exception:
            pass


# Import necessary functions from Phase 1 (keeping same logic)
def prepare_split_datasets(config):
    """Prepare train/val/test datasets"""
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
    """Evaluate policy on a specific split"""
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
                if len(trajectory_context['observations']) == 0:
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
    """Main Phase 2 optimization function"""
    print("🚀 Starting Optuna Phase 2: VAE and Training Dynamics Optimization")
    print("🏗️  Fixed Architecture: latent_dim=512, hidden_dim=1024 (from Phase 1)")
    print("🎯 Optimizing: VAE beta, context length, batch sizes, training dynamics")

    study_name = os.getenv("OPTUNA_STUDY", f"varibad_phase2_{int(time.time())}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=8,    # More startup trials for complex space
            n_warmup_steps=1500,   # More warmup for longer episodes
            interval_steps=600,    # Match val_interval
        ),
    )

    print(f"Study: {study_name}")
    print("Sampler: TPE with enhanced MedianPruner for longer training")

    # Phase 2 settings: fewer parallel jobs due to longer trials
    total_trials = int(os.getenv("TOTAL_TRIALS", "150"))  # More trials for complex space
    n_jobs = int(os.getenv("N_JOBS", "10"))  # Fewer parallel jobs

    print(f"Total trials: {total_trials}")
    print(f"Parallel jobs: {n_jobs}")
    print(f"Expected runtime: ~{total_trials * 45 / n_jobs / 60:.1f} hours")

    try:
        study.optimize(
            objective_phase2,
            n_trials=total_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
            gc_after_trial=True,
        )
    except KeyboardInterrupt:
        print("Optimization interrupted by user")

    # Results analysis
    print("\n" + "=" * 60)
    print("PHASE 2 OPTIMIZATION COMPLETE")
    print("=" * 60)

    print(f"Finished trials: {len(study.trials)}")
    print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

    if study.best_trial:
        print(f"\n🥇 Best Phase 2 Trial:")
        print(f"  Value: {study.best_trial.value:.6f}")
        print("  VAE Parameters:")
        for key in ['vae_beta', 'vae_update_freq', 'seq_len']:
            if key in study.best_trial.params:
                print(f"    {key}: {study.best_trial.params[key]}")
        print("  Training Parameters:")
        for key in ['batch_size', 'vae_batch_size', 'episodes_per_task', 'ppo_epochs']:
            if key in study.best_trial.params:
                print(f"    {key}: {study.best_trial.params[key]}")
        print("  Learning Rates:")
        for key in ['vae_lr', 'policy_lr']:
            if key in study.best_trial.params:
                print(f"    {key}: {study.best_trial.params[key]:.2e}")

    # Save results
    results_dir = Path("optuna_phase2_results")
    results_dir.mkdir(exist_ok=True)

    # Save all trials
    trials_data = []
    for tr in study.trials:
        row = {
            'trial_number': tr.number,
            'state': tr.state.name,
            'value': tr.value if tr.value is not None else 'N/A',
            'datetime_start': tr.datetime_start,
            'datetime_complete': tr.datetime_complete,
            'duration': tr.duration.total_seconds() if tr.duration else 'N/A',
        }
        for k, v in tr.params.items():
            row[f'param_{k}'] = v
        trials_data.append(row)

    if trials_data:
        import pandas as pd
        trials_df = pd.DataFrame(trials_data)
        trials_csv_path = results_dir / f"{study_name}_all_trials.csv"
        trials_df.to_csv(trials_csv_path, index=False)
        print(f"\n💾 All trials saved: {trials_csv_path}")

    # Save study object
    study_path = results_dir / f"{study_name}_study.pkl"
    with open(study_path, 'wb') as f:
        import pickle
        pickle.dump(study, f)

    # Save best parameters
    if study.best_trial:
        best_params_json = results_dir / f"{study_name}_best_params.json"
        with open(best_params_json, 'w') as f:
            json.dump(study.best_trial.params, f, indent=2)

        # Create final config file for production runs
        final_config = {
            # Fixed from Phase 1
            "latent_dim": 512,
            "hidden_dim": 1024,
            
            # Optimized in Phase 2
            **study.best_trial.params,
            
            # Base parameters
            "data_path": "environments/data/sp500_rl_ready_cleaned.parquet",
            "train_end": "2015-12-31",
            "val_end": "2020-12-31",
            "num_assets": 30,
            "device": "cuda",
            
            # Extended training for final runs
            "max_episodes": 10000,
            "val_interval": 1000,
            "val_episodes": 100,
            "test_episodes": 200,
            
            # Keep successful defaults
            "ppo_clip_ratio": 0.2,
            "value_loss_coef": 0.5,
            "max_grad_norm": 0.5,
            "gae_lambda": 0.95,
            "discount_factor": 0.99,
            
            "exp_name": f"final_optimized_phase2",
        }
        
        final_config_path = results_dir / "final_optimized_config.json"
        with open(final_config_path, 'w') as f:
            json.dump(final_config, f, indent=2)
        
        print(f"📋 Final optimized config saved: {final_config_path}")

    print(f"\n🎉 Phase 2 complete! Next steps:")
    print(f"   1. Run analysis: python analyze_phase2_results.py")
    print(f"   2. Final training: python main.py --config optuna_phase2_results/final_optimized_config.json")
    print(f"   3. Compare with Phase 1 baseline")


if __name__ == "__main__":
    main()