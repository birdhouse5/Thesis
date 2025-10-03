import optuna
import numpy as np
import torch

from config import ExperimentConfig, experiment_to_training_config
from main import prepare_environments, create_models
from algorithms.trainer import PPOTrainer

def objective(trial):
    # --- PPO hyperparams to tune ---
    policy_lr    = trial.suggest_loguniform("policy_lr", 1e-5, 3e-4)
    ppo_clip_ratio = trial.suggest_uniform("ppo_clip_ratio", 0.1, 0.3)
    entropy_coef = trial.suggest_uniform("entropy_coef", 0.0, 0.05)
    value_loss_coef = trial.suggest_uniform("value_loss_coef", 0.25, 1.0)
    gae_lambda   = trial.suggest_uniform("gae_lambda", 0.9, 0.98)
    latent_dim  = trial.suggest_categorical("latent_dim", [32, 64, 128])

    # --- Fixed encoder hyperparams ---
    exp = ExperimentConfig(seed=0, asset_class="crypto", encoder="vae")
    cfg = experiment_to_training_config(exp)

    cfg.latent_dim   = latent_dim
    cfg.hidden_dim   = 768
    cfg.vae_beta     = 0.0709877778524465
    cfg.policy_lr    = policy_lr
    cfg.ppo_clip_ratio = ppo_clip_ratio
    cfg.entropy_coef = entropy_coef
    cfg.value_loss_coef = value_loss_coef
    cfg.gae_lambda   = gae_lambda

    # --- Prepare envs and models ---
    envs, split_tensors, datasets = prepare_environments(cfg)
    train_env = envs["train"]
    obs_shape = (cfg.num_assets, len(split_tensors["train"]["feature_columns"]))
    vae, policy = create_models(cfg, obs_shape)
    trainer = PPOTrainer(train_env, policy, vae, cfg)

    # --- Training loop ---
    rewards = []
    max_episodes = 800  # keep small for HPO speed
    for ep in range(max_episodes):
        task = train_env.sample_task()
        train_env.set_task(task)

        result = trainer.train_episode()
        rewards.append(result["episode_sum_reward"])

        # pruning
        trial.report(np.mean(rewards[-20:]), step=ep)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(rewards[-50:]))

if __name__ == "__main__":
    study = optuna.create_study(
        study_name="ppo_only_hpo",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50)
    )
    study.optimize(objective, n_trials=100)
    print("Best params:", study.best_trial.params)
