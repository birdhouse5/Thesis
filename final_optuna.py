import optuna
import numpy as np
import torch
import argparse

from config import ExperimentConfig, experiment_to_training_config
from main import prepare_environments, create_models
from algorithms.trainer import PPOTrainer

def objective(trial, asset_class, reward_type):
    # --- Suggest hyperparams ---
    latent_dim   = trial.suggest_categorical("latent_dim", [16, 32, 64, 128])
    hidden_dim   = trial.suggest_categorical("hidden_dim", [256, 512, 768])
    vae_lr       = trial.suggest_loguniform("vae_lr", 1e-5, 1e-3)
    policy_lr    = trial.suggest_loguniform("policy_lr", 1e-5, 3e-4)
    vae_beta     = trial.suggest_loguniform("vae_beta", 1e-3, 1.0)
    entropy_coef = trial.suggest_uniform("entropy_coef", 0.0, 0.05)
    clip_ratio   = trial.suggest_uniform("ppo_clip_ratio", 0.1, 0.3)
    eta          = trial.suggest_loguniform("eta", 1e-3, 0.1)
    reward_lookback = trial.suggest_int("reward_lookback", 10, 50)

    # --- Config ---
    exp = ExperimentConfig(
        seed=0, 
        asset_class=asset_class, 
        encoder="vae"
    )
    cfg = experiment_to_training_config(exp)
    
    # Apply hyperparameters
    cfg.latent_dim   = latent_dim
    cfg.hidden_dim   = hidden_dim
    cfg.vae_lr       = vae_lr
    cfg.policy_lr    = policy_lr
    cfg.vae_beta     = vae_beta
    cfg.entropy_coef = entropy_coef
    cfg.ppo_clip_ratio = clip_ratio
    cfg.eta = eta
    cfg.reward_type = reward_type
    cfg.reward_lookback = reward_lookback

    # --- Prepare envs and models ---
    envs, split_tensors, datasets = prepare_environments(cfg)
    train_env = envs["train"]
    obs_shape = (cfg.num_assets, len(split_tensors["train"]["feature_columns"]))
    vae, policy = create_models(cfg, obs_shape)
    trainer = PPOTrainer(train_env, policy, vae, cfg)

    # --- Training loop ---
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_class", type=str, default="sp500", choices=["sp500", "crypto"])
    parser.add_argument("--reward_type", type=str, default="dsr", choices=["dsr", "sharpe", "drawdown"])
    parser.add_argument("--n_trials", type=int, default=50)
    args = parser.parse_args()

    study_name = f"ppo_vae_{args.asset_class}_{args.reward_type}"
    
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
    
    print(f"\n=== Best Trial ({args.reward_type}) ===")
    print(f"Value: {study.best_trial.value}")
    print(f"Params: {study.best_trial.params}")

    # sp500 dsr 1
    # crypto dsr 2



    # sp500 sharpe 4
# === Best Trial (sharpe) ===
# Value: -106.20400454998017
# Params: {'latent_dim': 32, 
# 'hidden_dim': 512, 
# 'vae_lr': 0.00032768209762336516, 
# 'policy_lr': 8.854562053336064e-05, 
# 'vae_beta': 0.0014277747724047083, 
# 'entropy_coef': 0.04356214051589346, 
# 'ppo_clip_ratio': 0.2983609586372239, 
# 'eta': 0.001069954956673335, 
# 'reward_lookback': 27}

    


    # sp500 drawdown 5
#     === Best Trial (drawdown) ===
# Value: 37.967431259155276
# Params: {'latent_dim': 16, 
# 'hidden_dim': 256, 
# 'vae_lr': 1.3731078963442245e-05, 
# 'policy_lr': 1.0374001123774155e-05, 
# 'vae_beta': 0.03462979100777297, 
# 'entropy_coef': 0.030010195766720024, 
# 'ppo_clip_ratio': 0.15292418418968373, 
# 'eta': 0.004665718933151942, 
# reward_lookback': 50}


# crypto drawdown 6