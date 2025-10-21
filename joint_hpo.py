# joint_hpo.py
import optuna
import numpy as np
import torch
import argparse
import json
from pathlib import Path

from config import ExperimentConfig, experiment_to_training_config
from main import prepare_environments, create_models
from algorithms.trainer import PPOTrainer
from hpo_utils import load_hpo_params


def objective(trial, asset_class, reward_type, vae_config_path, ppo_config_path):
    """
    Fine-tune interaction parameters using optimized VAE + PPO as foundation.
    
    Only optimizes parameters that affect VAE-PPO coupling.
    """
    
    # === Load baseline configs ===
    exp = ExperimentConfig(seed=0, asset_class=asset_class, encoder="vae")
    cfg = experiment_to_training_config(exp)
    
    # Apply optimized VAE params
    cfg = load_hpo_params(vae_config_path, cfg)
    # Apply optimized PPO params
    cfg = load_hpo_params(ppo_config_path, cfg)
    
    # === Fine-tune interaction parameters ===
    # VAE update frequency relative to policy updates
    cfg.vae_update_freq = trial.suggest_categorical("vae_update_freq", [1, 2, 3, 5])
    
    # Entropy affects trajectory diversity for VAE training
    cfg.entropy_coef = trial.suggest_uniform("entropy_coef", 
                                              max(0.0, cfg.entropy_coef - 0.01),
                                              min(0.05, cfg.entropy_coef + 0.01))
    
    # KL weight might need adjustment with new policy behavior
    cfg.vae_beta = trial.suggest_loguniform("vae_beta",
                                             cfg.vae_beta * 0.5,
                                             cfg.vae_beta * 2.0)
    
    # Learning rate balance
    lr_ratio = trial.suggest_uniform("lr_ratio", 0.5, 2.0)
    cfg.vae_lr = cfg.vae_lr * lr_ratio
    
    cfg.reward_type = reward_type
    
    # === Prepare envs and models ===
    envs, split_tensors, datasets = prepare_environments(cfg)
    train_env = envs["train"]
    obs_shape = (cfg.num_assets, len(split_tensors["train"]["feature_columns"]))
    vae, policy = create_models(cfg, obs_shape)
    trainer = PPOTrainer(train_env, policy, vae, cfg)
    
    # === Training loop (longer for fine-tuning) ===
    rewards = []
    max_tasks = 75  # More tasks for stability
    
    for task_idx in range(max_tasks):
        task = train_env.sample_task()
        train_env.set_task(task)
        result = trainer.train_on_task()
        rewards.append(result["task_total_reward"])
        
        # Optuna pruning
        trial.report(np.mean(rewards[-15:]), step=task_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return float(np.mean(rewards[-25:]))


def print_best_callback(study, trial):
    """Print current best parameters after each trial."""
    print(f"\n--- Trial {trial.number} complete ---")
    print(f"Current trial value: {trial.value}")
    print(f"Best value so far: {study.best_value}")
    print(f"Best params so far: {study.best_params}")
    print("-" * 50)


def save_best_params(study, asset_class, reward_type, vae_config, ppo_config):
    """Save best fine-tuned parameters."""
    output_dir = Path("hpo_results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"best_params_{asset_class}_{reward_type}_joint.json"
    
    # Load base configs
    with open(vae_config, 'r') as f:
        vae_data = json.load(f)
    with open(ppo_config, 'r') as f:
        ppo_data = json.load(f)
    
    # Merge all parameters
    final_params = {**vae_data['best_params'], **ppo_data['best_params']}
    
    # Override with fine-tuned interaction params
    final_params.update(study.best_trial.params)
    
    result = {
        "study_name": study.study_name,
        "asset_class": asset_class,
        "reward_type": reward_type,
        "encoder": "vae",
        "optimization_stages": {
            "stage1_vae": {
                "value": vae_data['best_value'],
                "n_trials": vae_data['n_trials']
            },
            "stage2_ppo": {
                "value": ppo_data['best_value'],
                "n_trials": ppo_data['n_trials']
            },
            "stage3_joint": {
                "value": study.best_trial.value,
                "n_trials": len(study.trials)
            }
        },
        "best_params": final_params,
        "fine_tuned_params": study.best_trial.params,
        "note": "3-stage optimization: VAE → PPO → Joint fine-tuning"
    }
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Joint fine-tuned parameters saved to {output_file}")
    print(f"\nFine-tuned interaction parameters:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_class", type=str, required=True, choices=["sp500", "crypto"])
    parser.add_argument("--reward_type", type=str, required=True, choices=["dsr", "sharpe", "drawdown"])
    parser.add_argument("--vae_config", type=str, required=True,
                       help="Path to VAE HPO results JSON")
    parser.add_argument("--ppo_config", type=str, required=True,
                       help="Path to PPO HPO results JSON")
    parser.add_argument("--n_trials", type=int, default=30)
    args = parser.parse_args()
    
    study_name = f"joint_finetune_{args.asset_class}_{args.reward_type}"
    
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=15)
    )
    
    study.optimize(
        lambda trial: objective(trial, args.asset_class, args.reward_type, 
                               args.vae_config, args.ppo_config),
        n_trials=args.n_trials,
        callbacks=[print_best_callback]
    )
    
    print(f"\n=== Best Trial (Joint Fine-tuning) ===")
    print(f"Value: {study.best_trial.value}")
    print(f"Fine-tuned params: {study.best_trial.params}")
    
    save_best_params(study, args.asset_class, args.reward_type, 
                    args.vae_config, args.ppo_config)