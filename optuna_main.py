import torch
torch.set_num_threads(1)  # keep CPU usage per trial predictable
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True   # speed on Ampere+
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False         # avoid huge autotune spikes
    torch.backends.cudnn.deterministic = False
    
import optuna
import logging
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, Any

# Import your existing modules
from environments.dataset import create_split_datasets
from environments.env import MetaEnv
from models.vae import VAE
from models.policy import PortfolioPolicy
from algorithms.trainer import PPOTrainer
from run_logger import RunLogger, seed_everything
import os

# Optuna-specific settings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptunaConfig:
    """Configuration class for Optuna trials"""
    def __init__(self, trial: optuna.Trial):
        # Fixed base parameters (not optimized in Phase 1)
        self.data_path = "environments/data/sp500_rl_ready_cleaned.parquet"
        self.train_end = '2015-12-31'
        self.val_end = '2020-12-31'
        self.num_assets = 30
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Fixed environment parameters
        self.seq_len = 60
        self.min_horizon = 45
        self.max_horizon = 60
        
        # Fixed training parameters (reasonable defaults)
        self.max_episodes = 4000  # Reduced for faster trials
        self.episodes_per_task = 5
        self.batch_size = 4096  # Memory-safe for 24GB
        self.vae_batch_size = 1024
        
        # Fixed PPO parameters
        self.ppo_epochs = 4
        self.ppo_clip_ratio = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.gae_lambda = 0.95
        self.discount_factor = 0.99
        
        # Fixed VAE parameters
        self.vae_beta = 0.1
        self.vae_update_freq = 1
        
        # Fixed validation parameters
        self.val_interval = 500
        self.val_episodes = 50  # Reduced for faster validation
        self.test_episodes = 50
        self.log_interval = 100
        self.save_interval = 1000
        
        # PHASE 1: Optimize Architecture + Learning Rates
        # Architecture parameters (ranges from generate_configs.py)
        self.latent_dim = trial.suggest_categorical('latent_dim', [64, 128, 256, 512])
        self.hidden_dim = trial.suggest_categorical('hidden_dim', [256, 512, 1024, 2048])
        
        # Learning rate parameters
        self.vae_lr = trial.suggest_float('vae_lr', 1e-5, 3e-3, log=True)
        self.policy_lr = trial.suggest_float('policy_lr', 1e-5, 3e-3, log=True)
        
        # Generate experiment name
        self.exp_name = f"optuna_t{trial.number}_l{self.latent_dim}_h{self.hidden_dim}"
        
        # Seed for reproducibility
        self.seed = 42
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def update_vae_fixed(trainer):
    """Fixed VAE update with proper gradient handling"""
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

    # Backprop through the accumulated tensor loss
    avg_loss = total_loss / loss_count
    avg_loss.backward()
    torch.nn.utils.clip_grad_norm_(trainer.vae.parameters(), trainer.config.max_grad_norm)
    trainer.vae_optimizer.step()
    
    return float(avg_loss.item())

# --- GPU cache helper ---
def _free_cuda_cache(tag: str = ""):
    """Release unreferenced CUDA memory back to the driver + Python GC."""
    import torch, gc, time
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()
    time.sleep(0.02)  # tiny pause to let the allocator settle


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function for Phase 1 optimization.
    Returns validation Sharpe ratio to maximize.
    """
    import gc, time, torch
    try:
        # Create configuration for this trial (kept exactly as in your code)
        config = OptunaConfig(trial)  # ← you instantiate the config here; preserved

        # Slight per-trial seed offset for stable parallelism (keeps your base seed)
        seed_everything(config.seed + int(trial.number))

        # Set up run directory and logging
        run_dir = Path("optuna_runs") / f"trial_{trial.number}_{int(time.time())}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run = RunLogger(run_dir, config.to_dict(), name=config.exp_name)

        logger.info(f"Trial {trial.number}: {config.exp_name}")
        logger.info(
            f"Params: latent_dim={config.latent_dim}, hidden_dim={config.hidden_dim}, "
            f"vae_lr={config.vae_lr:.2e}, policy_lr={config.policy_lr:.2e}"
        )

        # ------- dataset/env prep (CPU) -------
        try:
            # Prepare datasets
            split_tensors, split_datasets = prepare_split_datasets(config)

            # Create environments
            train_env = create_meta_env(
                split_tensors['train'], split_tensors['train']['feature_columns'], config
            )
            val_env = create_meta_env(
                split_tensors['val'], split_tensors['val']['feature_columns'], config
            )

            # ------- OOM GUARD (GPU-heavy section) -------
            try:
                # Get observation shape
                task = train_env.sample_task()
                train_env.set_task(task)
                initial_obs = train_env.reset()
                obs_shape = initial_obs.shape

                # Initialize models
                vae, policy = initialize_models(
                    config, obs_shape, split_tensors['train']['feature_columns']
                )

                # Initialize trainer
                trainer = PPOTrainer(env=train_env, policy=policy, vae=vae, config=config)

                # CRITICAL FIX (kept): replace broken VAE update
                trainer.update_vae = lambda: update_vae_fixed(trainer)

                # Training loop with early stopping (your logic, unchanged)
                episodes_trained = 0
                best_val_sharpe = float('-inf')
                patience = 3
                no_improvement_count = 0

                logger.info(f"Starting training for trial {trial.number}")

                while episodes_trained < config.max_episodes:
                    task = train_env.sample_task()
                    train_env.set_task(task)

                    for _ in range(config.episodes_per_task):
                        episode_result = trainer.train_episode()

                        # Log training metrics
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

                            # Log validation
                            run.log_val(
                                episodes_trained,
                                sharpe=current_val_sharpe,
                                reward=current_val_sharpe,
                                cum_wealth=val_results['avg_return'],
                            )

                            logger.info(
                                f"Trial {trial.number}, Episode {episodes_trained}: "
                                f"val_sharpe={current_val_sharpe:.4f}"
                            )

                            # Early stopping + save best
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
                                    },
                                    save_path,
                                )
                            else:
                                no_improvement_count += 1

                            # Report to Optuna / pruning
                            trial.report(current_val_sharpe, episodes_trained)
                            if trial.should_prune():
                                logger.info(f"Trial {trial.number} pruned at episode {episodes_trained}")
                                # close run before pruning
                                try: run.close()
                                except Exception: pass
                                raise optuna.TrialPruned()

                            if no_improvement_count >= patience:
                                logger.info(f"Trial {trial.number} early stopped due to no improvement")
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
                    f"CUDA OOM on trial {trial.number}; pruning this trial. "
                    f"(batch_size={getattr(config,'batch_size',None)}, "
                    f"hidden_dim={getattr(config,'hidden_dim',None)}, "
                    f"latent_dim={getattr(config,'latent_dim',None)})"
                )
                _free_cuda_cache("oom")
                try: run.close()
                except Exception: pass
                raise optuna.TrialPruned() from e

            except RuntimeError as e:
                # Some OOMs surface as generic RuntimeError with 'out of memory'
                if "out of memory" in str(e).lower():
                    logger.warning(f"RuntimeError OOM on trial {trial.number}; pruning.")
                    _free_cuda_cache("oom")
                    try: run.close()
                    except Exception: pass
                    raise optuna.TrialPruned() from e
                raise  # not an OOM → bubble up

        finally:
            # Aggressive cleanup regardless of success/exception in the middle block
            for name in ("trainer", "vae", "policy", "train_env", "val_env",
                         "split_tensors", "split_datasets", "initial_obs"):
                if name in locals():
                    del locals()[name]
            logger.info(
                f"[mem before free] allocated={torch.cuda.memory_allocated()/1e6:.0f}MB "
                f"reserved={torch.cuda.memory_reserved()/1e6:.0f}MB"
            )
            _free_cuda_cache("finally")
            logger.info(
                f"[mem after free]  allocated={torch.cuda.memory_allocated()/1e6:.0f}MB "
                f"reserved={torch.cuda.memory_reserved()/1e6:.0f}MB"
            )

    except optuna.TrialPruned:
        # already handled/logged; just propagate
        raise

    except Exception as e:
        # catch-all for anything before/after the inner try/finally
        import traceback
        logger.error(f"Trial {trial.number} failed with exception: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise

    finally:
        # Ensure run logger is closed even if failure happened very early
        try:
            if 'run' in locals():
                run.close()
        except Exception:
            pass




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
    """Main Optuna optimization function"""
    import os
    print("🚀 Starting Optuna Phase 1: Architecture + Learning Rate Optimization")

    # In-memory study (no SQLite). We still keep a readable name for logs/files.
    study_name = os.getenv("OPTUNA_STUDY", f"varibad_phase1_{int(time.time())}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1000,
            interval_steps=500,
        ),
    )

    print(f"Study created: {study_name}")
    print("Storage: In-memory (no SQLite/Postgres)")
    print("Objective: Maximize validation Sharpe ratio")
    print("Sampler: TPE with MedianPruner")

    # Total trials & parallel threads (defaults chosen for your single-VM setup)
    total_trials = int(os.getenv("TOTAL_TRIALS", os.getenv("TRIALS_PER_WORKER", "100")))
    n_jobs = int(os.getenv("N_JOBS", "15"))  # 15 parallel trials in one process

    try:
        study.optimize(
            objective,
            n_trials=total_trials,
            n_jobs=n_jobs,                 # ← fan out threads (single process)
            show_progress_bar=True,
            gc_after_trial=True,
        )
    except KeyboardInterrupt:
        print("Optimization interrupted by user")

    # ===== Results / exports (as in your file) =====
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

    if study.best_trial:
        print("\nBest trial:")
        print(f"  Value: {study.best_trial.value:.6f}")
        print("  Params:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")

    # Save results to CSV / JSON / Pickle (kept the same pattern)
    results_dir = Path("optuna_results")
    results_dir.mkdir(exist_ok=True)

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

    trials_csv_path = None
    if trials_data:
        import pandas as pd
        trials_df = pd.DataFrame(trials_data)
        trials_csv_path = results_dir / f"{study_name}_all_trials.csv"
        trials_df.to_csv(trials_csv_path, index=False)
        print(f"All trials saved: {trials_csv_path}")

    study_path = results_dir / f"{study_name}_study.pkl"
    with open(study_path, 'wb') as f:
        import pickle
        pickle.dump(study, f)

    best_params_json = best_trial_csv = None
    if study.best_trial:
        best_params_json = results_dir / f"{study_name}_best_params.json"
        with open(best_params_json, 'w') as f:
            json.dump(study.best_trial.params, f, indent=2)

        import pandas as pd
        best_trial_csv = results_dir / f"{study_name}_best_trial.csv"
        best_data = [{
            'trial_number': study.best_trial.number,
            'best_value': study.best_trial.value,
            **study.best_trial.params,
        }]
        pd.DataFrame(best_data).to_csv(best_trial_csv, index=False)

    print(f"\nResults saved to {results_dir}")
    if trials_csv_path: print(f"All trials CSV: {trials_csv_path}")
    if best_params_json: print(f"Best params JSON: {best_params_json}")
    if best_trial_csv: print(f"Best trial CSV: {best_trial_csv}")
    print(f"Study pickle: {study_path}")

    try:
        import plotly
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(results_dir / f"{study_name}_history.html")
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_html(results_dir / f"{study_name}_importance.html")
        print("Plots saved: history.html, importance.html")
    except ImportError:
        print("Plotly not available - skipping plots")


if __name__ == "__main__":
    main()
