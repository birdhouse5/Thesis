import optuna
import numpy as np
import argparse
import json
from pathlib import Path

from config import ExperimentConfig, experiment_to_training_config
from main import prepare_environments, create_models
from algorithms.trainer import PPOTrainer
from hpo_utils import load_hpo_params


def objective(trial, asset_class, reward_type, encoder, base_params_path=None):
    """Optimize concentration penalty parameters."""
    
    # === Parameters to Optimize ===
    concentration_target = trial.suggest_uniform("concentration_target", 0.01, 0.2)
    concentration_lambda = trial.suggest_loguniform("concentration_lambda", 0.01, 1.0)
    
    # === Config ===
    exp = ExperimentConfig(
        seed=0,
        asset_class=asset_class,
        encoder=encoder,
        concentration_penalty=True
    )
    cfg = experiment_to_training_config(exp)
    
    # Load base parameters if provided
    if base_params_path:
        cfg = load_hpo_params(base_params_path, cfg)
    
    # Apply concentration parameters
    cfg.concentration_target = concentration_target
    cfg.concentration_lambda = concentration_lambda
    cfg.concentration_penalty = True
    cfg.reward_type = reward_type
    
    # === Training ===
    envs, split_tensors, datasets = prepare_environments(cfg)
    train_env = envs["train"]
    obs_shape = (cfg.num_assets, len(split_tensors["train"]["feature_columns"]))
    vae, policy = create_models(cfg, obs_shape)
    trainer = PPOTrainer(train_env, policy, vae, cfg)
    
    rewards = []
    max_tasks = 50
    
    for task_idx in range(max_tasks):
        task = train_env.sample_task()
        train_env.set_task(task)
        result = trainer.train_on_task()
        rewards.append(result["task_total_reward"])
        
        trial.report(np.mean(rewards[-10:]), step=task_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return float(np.mean(rewards[-20:]))


def save_best_params(study, asset_class, reward_type, encoder):
    output_dir = Path("hpo_results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"best_params_{asset_class}_{reward_type}_{encoder}_herfindahl.json"
    
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
    parser.add_argument("--asset_class", type=str, default="sp500")
    parser.add_argument("--reward_type", type=str, default="dsr")
    parser.add_argument("--encoder", type=str, default="vae")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--load_base_params", type=str, default=None)
    args = parser.parse_args()
    
    study = optuna.create_study(
        study_name=f"herfindahl_{args.asset_class}_{args.reward_type}_{args.encoder}",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner()
    )
    
    study.optimize(
        lambda trial: objective(trial, args.asset_class, args.reward_type, args.encoder, args.load_base_params),
        n_trials=args.n_trials
    )
    
    save_best_params(study, args.asset_class, args.reward_type, args.encoder)