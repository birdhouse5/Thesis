import optuna
import numpy as np
import torch

from config import ExperimentConfig, experiment_to_training_config
from main import prepare_environments, create_models
from algorithms.trainer import PPOTrainer

# === Objective function ===
def objective(trial):

    # --- Suggest hyperparams ---
    latent_dim   = trial.suggest_categorical("latent_dim", [16, 32, 64, 128])
    hidden_dim   = trial.suggest_categorical("hidden_dim", [256, 512, 768])
    vae_lr       = trial.suggest_loguniform("vae_lr", 1e-5, 1e-3)
    policy_lr    = trial.suggest_loguniform("policy_lr", 1e-5, 3e-4)
    vae_beta     = trial.suggest_loguniform("vae_beta", 1e-3, 1.0)
    entropy_coef = trial.suggest_uniform("entropy_coef", 0.0, 0.05)
    clip_ratio   = trial.suggest_uniform("ppo_clip_ratio", 0.1, 0.3)

    # --- Base experiment config ---
    exp = ExperimentConfig(seed=0, asset_class="sp500", encoder="vae")
    cfg = experiment_to_training_config(exp)

    # Override with trial params
    cfg.latent_dim   = latent_dim
    cfg.hidden_dim   = hidden_dim
    cfg.vae_lr       = vae_lr
    cfg.policy_lr    = policy_lr
    cfg.vae_beta     = vae_beta
    cfg.entropy_coef = entropy_coef
    cfg.ppo_clip_ratio = clip_ratio

    # --- Prepare envs ---
    envs, split_tensors, datasets = prepare_environments(cfg)
    train_env = envs["train"]

    # --- Create models ---
    obs_shape = (cfg.num_assets, len(split_tensors["train"]["feature_columns"]))
    vae, policy = create_models(cfg, obs_shape)

    trainer = PPOTrainer(train_env, policy, vae, cfg)

    # --- Training loop (shortened for HPO) ---
    rewards = []
    max_episodes = 200   # keep short for trial
    for ep in range(max_episodes):
        result = trainer.train_episode()
        rewards.append(result["episode_sum_reward"])

        # report to Optuna for pruning
        trial.report(np.mean(rewards[-20:]), step=ep)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # --- Return mean of last 50 episodes ---
    return float(np.mean(rewards[-50:]))

# === Run study ===
if __name__ == "__main__":
    study = optuna.create_study(
        study_name="ppo_vae_portfolio",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50)
    )
    study.optimize(objective, n_trials=50)
    print("Best params:", study.best_trial.params)
