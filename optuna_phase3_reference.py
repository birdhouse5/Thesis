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

class OptunaPhase3Config:
    """Phase 3 Configuration - Early Stopping Optimization"""
    def __init__(self, trial: optuna.Trial):
        # FIXED from Phase 2 Trial 46 (all optimal parameters)
        self.latent_dim = 512
        self.hidden_dim = 1024
        self.vae_lr = 0.0010748206602172
        self.policy_lr = 0.0020289998766945
        self.vae_beta = 0.0125762666385515
        self.vae_update_freq = 5
        #self.seq_len = trial.suggest_categorical('seq_len', [120, 150, 200])
        self.seq_len = 120
        self.episodes_per_task = 3
        self.batch_size = 8192
        self.vae_batch_size = 1024
        self.ppo_epochs = 8
        self.entropy_coef = 0.0013141391952945
        
        # PHASE 3: Optimize early stopping and episode length
        self.max_episodes = trial.suggest_categorical('max_episodes', [2000, 4000, 6000, 8000])
        self.early_stopping_patience = trial.suggest_categorical('early_stopping_patience', [3, 5, 8, 12])
        self.early_stopping_min_delta = trial.suggest_categorical('early_stopping_min_delta', [0.001, 0.01, 0.05])
        self.val_interval = trial.suggest_categorical('val_interval', [200, 300, 500])
        
        # Derived early stopping settings
        self.min_episodes_before_stopping = max(1000, self.max_episodes // 4)  # Don't stop too early
        
        # Fixed base parameters
        self.data_path = "environments/data/sp500_rl_ready_cleaned.parquet"
        self.train_end = '2015-12-31'
        self.val_end = '2020-12-31'
        self.num_assets = 30
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Derived horizons
        self.max_horizon = min(self.seq_len - 10, int(self.seq_len * 0.8))
        self.min_horizon = max(self.max_horizon - 15, self.max_horizon // 2)
        
        # Validation settings
        self.val_episodes = 50  # Smaller for faster trials
        self.test_episodes = 100
        self.log_interval = 200
        self.save_interval = 2000
        
        # Fixed PPO parameters (from Phase 2)
        self.ppo_clip_ratio = 0.2
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.gae_lambda = 0.95
        self.discount_factor = 0.99
        
        # Experiment naming
        self.exp_name = f"phase3_t{trial.number}_ep{self.max_episodes}_pat{self.early_stopping_patience}"
        
        # Seed for reproducibility
        self.seed = 42
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def update_vae_fixed(trainer):
    """Fixed VAE update from previous phases"""
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


def objective_phase3(trial: optuna.Trial) -> float:
    """
    Phase 3 objective function focusing on early stopping optimization.
    Returns validation Sharpe ratio from early-stopped model.
    """
    import gc, time, torch
    try:
        # Create Phase 3 configuration
        config = OptunaPhase3Config(trial)
        
        # Per-trial seed offset
        seed_everything(config.seed + int(trial.number))

        # Set up run directory
        run_dir = Path("optuna_phase3_runs") / f"trial_{trial.number}_{int(time.time())}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run = RunLogger(run_dir, config.to_dict(), name=config.exp_name)

        logger.info(f"Phase 3 Trial {trial.number}: {config.exp_name}")
        logger.info(f"Early stopping: max_episodes={config.max_episodes}, patience={config.early_stopping_patience}")
        logger.info(f"Validation: interval={config.val_interval}, min_delta={config.early_stopping_min_delta}")

        try:
            # Prepare datasets (same as previous phases)
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

                # Initialize trainer with early stopping
                trainer = PPOTrainer(env=train_env, policy=policy, vae=vae, config=config)
                trainer.update_vae = lambda: update_vae_fixed(trainer)

                # Training loop with early stopping
                episodes_trained = 0
                best_val_sharpe = float('-inf')
                final_val_sharpe = None

                logger.info(f"Starting Phase 3 training for trial {trial.number}")

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
                            final_val_sharpe = current_val_sharpe  # Track latest

                            run.log_val(
                                episodes_trained,
                                sharpe=current_val_sharpe,
                                reward=current_val_sharpe,
                                cum_wealth=val_results['avg_return'],
                            )

                            logger.info(f"Trial {trial.number}, Episode {episodes_trained}: val_sharpe={current_val_sharpe:.4f}")

                            # Early stopping check
                            should_stop = trainer.add_validation_score(current_val_sharpe)
                            
                            # Update best score
                            if current_val_sharpe > best_val_sharpe:
                                best_val_sharpe = current_val_sharpe
                                
                                # Save best model
                                save_path = run_dir / "best_model.pt"
                                torch.save({
                                    'trial_number': trial.number,
                                    'episodes_trained': episodes_trained,
                                    'vae_state_dict': vae.state_dict(),
                                    'policy_state_dict': policy.state_dict(),
                                    'best_val_sharpe': best_val_sharpe,
                                    'config': config.to_dict(),
                                    'early_stopping_state': trainer.get_early_stopping_state()
                                }, save_path)

                            # Report to Optuna
                            trial.report(current_val_sharpe, episodes_trained)
                            
                            # Check for pruning
                            if trial.should_prune():
                                logger.info(f"Trial {trial.number} pruned at episode {episodes_trained}")
                                try: run.close()
                                except Exception: pass
                                raise optuna.TrialPruned()

                            # Early stopping
                            if should_stop:
                                logger.info(f"Trial {trial.number} early stopped at episode {episodes_trained}")
                                break

                        if episodes_trained >= config.max_episodes or trainer.should_stop_early():
                            break

                    if episodes_trained >= config.max_episodes or trainer.should_stop_early():
                        break

                # Get final result
                early_stopping_state = trainer.get_early_stopping_state()
                final_score = early_stopping_state['best_val_score']
                
                if final_score == float('-inf'):
                    # No early stopping occurred, use last validation
                    final_score = final_val_sharpe if final_val_sharpe is not None else 0.0

                logger.info(f"Trial {trial.number} completed:")
                logger.info(f"  Episodes: {episodes_trained}")
                logger.info(f"  Early stopped: {early_stopping_state['early_stopped']}")
                logger.info(f"  Final score: {final_score:.4f}")
                
                run.close()
                return final_score

            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"CUDA OOM on trial {trial.number}; pruning.")
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


# Import functions from previous phases (same logic)
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
    """Main Phase 3 optimization function"""
    print("🚀 Starting Optuna Phase 3: Early Stopping Optimization")
    print("🎯 Fixed: All Phase 2 optimal parameters")
    print("🔧 Optimizing: Episode length + early stopping criteria")

    study_name = os.getenv("OPTUNA_STUDY", f"varibad_phase3_{int(time.time())}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1000,
            interval_steps=400,
        ),
    )

    print(f"Study: {study_name}")
    print("Focus: Finding optimal stopping point to prevent overfitting")

    # Phase 3 settings: smaller search space, more focused
    total_trials = int(os.getenv("TOTAL_TRIALS", "80"))  # Smaller space = fewer trials needed
    n_jobs = int(os.getenv("N_JOBS", "12"))

    print(f"Total trials: {total_trials}")
    print(f"Parallel jobs: {n_jobs}")
    print(f"Expected runtime: ~{total_trials * 25 / n_jobs / 60:.1f} hours")

    try:
        study.optimize(
            objective_phase3,
            n_trials=total_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
            gc_after_trial=True,
        )
    except KeyboardInterrupt:
        print("Optimization interrupted by user")

    # Results analysis
    print("\n" + "=" * 60)
    print("PHASE 3 OPTIMIZATION COMPLETE")
    print("=" * 60)

    print(f"Finished trials: {len(study.trials)}")
    print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

    if study.best_trial:
        print(f"\n🏆 Best Phase 3 Trial:")
        print(f"  Value: {study.best_trial.value:.6f}")
        print("  Early Stopping Parameters:")
        for key in ['max_episodes', 'early_stopping_patience', 'early_stopping_min_delta', 'val_interval']:
            if key in study.best_trial.params:
                print(f"    {key}: {study.best_trial.params[key]}")

    # Save results
    results_dir = Path("optuna_phase3_results")
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

        # Create final optimized config
        final_config = {
            # Fixed from Phase 1+2
            "latent_dim": 512,
            "hidden_dim": 1024,
            "vae_lr": 0.0010748206602172,
            "policy_lr": 0.0020289998766945,
            "vae_beta": 0.0125762666385515,
            "vae_update_freq": 5,
            "seq_len": 120,
            "episodes_per_task": 3,
            "batch_size": 8192,
            "vae_batch_size": 1024,
            "ppo_epochs": 8,
            "entropy_coef": 0.0013141391952945,
            
            # Optimized in Phase 3
            **study.best_trial.params,
            
            # Base parameters
            "data_path": "environments/data/sp500_rl_ready_cleaned.parquet",
            "train_end": "2015-12-31",
            "val_end": "2020-12-31",
            "num_assets": 30,
            "device": "cuda",
            
            # Final production settings
            "test_episodes": 200,
            "exp_name": "final_phase3_optimized",
        }
        
        final_config_path = results_dir / "final_phase3_config.json"
        with open(final_config_path, 'w') as f:
            json.dump(final_config, f, indent=2)
        
        print(f"📋 Final Phase 3 config saved: {final_config_path}")

    print(f"\n🎉 Phase 3 complete! Next steps:")
    print(f"   1. Analyze results: python analyze_phase3_results.py")
    print(f"   2. Final validation: python validate_best_model.py --model_path optuna_phase3_results/...")
    print(f"   3. Compare all 3 phases for thesis")


if __name__ == "__main__":
    main()