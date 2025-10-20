import optuna
import numpy as np
import torch
import argparse
import json
from pathlib import Path

from config import ExperimentConfig, experiment_to_training_config
from main import prepare_environments, create_models
from algorithms.trainer import PPOTrainer


def objective(trial, asset_class, reward_type, encoder):
    """Optimize only PPO parameters with fixed encoder architecture."""
    
    # === PPO Parameters to Optimize ===
    policy_lr = trial.suggest_loguniform("policy_lr", 1e-5, 3e-4)
    ppo_clip_ratio = trial.suggest_uniform("ppo_clip_ratio", 0.1, 0.3)
    entropy_coef = trial.suggest_uniform("entropy_coef", 0.0, 0.05)
    ppo_epochs = trial.suggest_int("ppo_epochs", 4, 16)
    value_loss_coef = trial.suggest_uniform("value_loss_coef", 0.3, 2.0)
    max_grad_norm = trial.suggest_uniform("max_grad_norm", 0.3, 2.0)
    gae_lambda = trial.suggest_uniform("gae_lambda", 0.90, 0.99)
    discount_factor = trial.suggest_uniform("discount_factor", 0.95, 0.999)
    ppo_minibatch_size = trial.suggest_categorical("ppo_minibatch_size", [64, 128, 256])
    
    # === Config ===
    exp = ExperimentConfig(
        seed=0,
        asset_class=asset_class,
        encoder=encoder
    )
    cfg = experiment_to_training_config(exp)
    
    # Apply PPO hyperparameters
    cfg.policy_lr = policy_lr
    cfg.ppo_clip_ratio = ppo_clip_ratio
    cfg.entropy_coef = entropy_coef
    cfg.ppo_epochs = ppo_epochs
    cfg.value_loss_coef = value_loss_coef
    cfg.max_grad_norm = max_grad_norm
    cfg.gae_lambda = gae_lambda
    cfg.discount_factor = discount_factor
    cfg.ppo_minibatch_size = ppo_minibatch_size
    cfg.reward_type = reward_type
    
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


def save_best_params(study, asset_class, reward_type, encoder):
    """Save best parameters to JSON file."""
    output_dir = Path("hpo_results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"best_params_{asset_class}_{reward_type}_{encoder}_ppo_only.json"
    
    result = {
        "study_name": study.study_name,
        "asset_class": asset_class,
        "reward_type": reward_type,
        "encoder": encoder,
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
    parser.add_argument("--encoder", type=str, default="vae", choices=["vae", "hmm", "none"])
    parser.add_argument("--n_trials", type=int, default=50)
    args = parser.parse_args()
    
    study_name = f"ppo_only_{args.asset_class}_{args.reward_type}_{args.encoder}"
    
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    study.optimize(
        lambda trial: objective(trial, args.asset_class, args.reward_type, args.encoder),
        n_trials=args.n_trials,
        callbacks=[print_best_callback]
    )
    
    print(f"\n=== Best Trial (PPO-only for {args.encoder}) ===")
    print(f"Value: {study.best_trial.value}")
    print(f"Params: {study.best_trial.params}")
    
    save_best_params(study, args.asset_class, args.reward_type, args.encoder)