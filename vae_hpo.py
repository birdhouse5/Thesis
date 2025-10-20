import optuna
import numpy as np
import torch
import argparse
import json
from pathlib import Path

from config import ExperimentConfig, experiment_to_training_config
from main import prepare_environments, create_models
from algorithms.trainer import PPOTrainer


def objective(trial, asset_class, reward_type):
    """Optimize only VAE parameters with PPO at defaults."""
    
    # === VAE Parameters to Optimize ===
    latent_dim = trial.suggest_categorical("latent_dim", [16, 32, 64, 128])
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768])
    vae_lr = trial.suggest_loguniform("vae_lr", 1e-5, 1e-3)
    vae_beta = trial.suggest_loguniform("vae_beta", 1e-3, 1.0)
    vae_update_freq = trial.suggest_categorical("vae_update_freq", [1, 3, 5, 10])
    vae_num_elbo_terms = trial.suggest_categorical("vae_num_elbo_terms", [4, 8, 16, 32])
    
    # === Config with VAE encoder ===
    exp = ExperimentConfig(
        seed=0,
        asset_class=asset_class,
        encoder="vae"
    )
    cfg = experiment_to_training_config(exp)
    
    # Apply VAE hyperparameters
    cfg.latent_dim = latent_dim
    cfg.hidden_dim = hidden_dim
    cfg.vae_lr = vae_lr
    cfg.vae_beta = vae_beta
    cfg.vae_update_freq = vae_update_freq
    cfg.vae_num_elbo_terms = vae_num_elbo_terms
    cfg.reward_type = reward_type
    
    # PPO stays at defaults from config.py
    
    # === Prepare envs and models ===
    envs, split_tensors, datasets = prepare_environments(cfg)
    train_env = envs["train"]
    obs_shape = (cfg.num_assets, len(split_tensors["train"]["feature_columns"]))
    vae, policy = create_models(cfg, obs_shape)
    trainer = PPOTrainer(train_env, policy, vae, cfg)
    
    # === Training loop ===
    rewards = []
    max_tasks = 50  # Reduced for HPO
    
    for task_idx in range(max_tasks):
        task = train_env.sample_task()
        train_env.set_task(task)
        result = trainer.train_on_task()
        rewards.append(result["task_total_reward"])
        
        # Optuna pruning
        trial.report(np.mean(rewards[-10:]), step=task_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return float(np.mean(rewards[-20:]))


def print_best_callback(study, trial):
    """Print current best parameters after each trial."""
    print(f"\n--- Trial {trial.number} complete ---")
    print(f"Current trial value: {trial.value}")
    print(f"Best value so far: {study.best_value}")
    print(f"Best params so far: {study.best_params}")
    print("-" * 50)


def save_best_params(study, asset_class, reward_type):
    """Save best parameters to JSON file."""
    output_dir = Path("hpo_results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"best_params_{asset_class}_{reward_type}_vae_only.json"
    
    result = {
        "study_name": study.study_name,
        "asset_class": asset_class,
        "reward_type": reward_type,
        "encoder": "vae",
        "best_value": study.best_trial.value,
        "best_params": study.best_trial.params,
        "n_trials": len(study.trials)
    }
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Best parameters saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_class", type=str, default="sp500", choices=["sp500", "crypto"])
    parser.add_argument("--reward_type", type=str, default="dsr", choices=["dsr", "sharpe", "drawdown"])
    parser.add_argument("--n_trials", type=int, default=50)
    args = parser.parse_args()
    
    study_name = f"vae_only_{args.asset_class}_{args.reward_type}"
    
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    study.optimize(
        lambda trial: objective(trial, args.asset_class, args.reward_type),
        n_trials=args.n_trials,
        callbacks=[print_best_callback]
    )
    
    print(f"\n=== Best Trial (VAE-only) ===")
    print(f"Value: {study.best_trial.value}")
    print(f"Params: {study.best_trial.params}")
    
    save_best_params(study, args.asset_class, args.reward_type)