# hpo.py
import os
if os.getenv("DISABLE_MLFLOW", "false").lower() == "true":
    import mlflow
    mlflow.start_run = lambda *a, **kw: open("/dev/null","w")  # dummy context
    mlflow.log_param = lambda *a, **kw: None
    mlflow.log_metric = lambda *a, **kw: None
    mlflow.set_tracking_uri = lambda *a, **kw: None


import argparse
from datetime import datetime
import json
import sys
from typing import List

import optuna
import mlflow

# Local imports from your repo
from config import ExperimentConfig, experiment_to_training_config
from main import run_training

def parse_args():
    p = argparse.ArgumentParser(description="Minimal Optuna HPO for variBAD-style Meta-RL")
    p.add_argument("--study-name", type=str, default="variBAD-HPO")
    p.add_argument("--storage", type=str, default=None, help="e.g. sqlite:///hpo.db (optional)")
    p.add_argument("--asset-class", type=str, choices=["sp500", "crypto"], required=True)
    p.add_argument("--encoder", type=str, choices=["vae", "none", "hmm"], default="vae")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--seeds", type=str, default="0", help="comma-separated list, e.g. '0,1,2'")
    p.add_argument("--max-episodes", type=int, default=1800, help="shorten for faster HPO")
    p.add_argument("--val-episodes", type=int, default=30, help="shorten for faster HPO")
    p.add_argument("--test-episodes", type=int, default=50, help="shorten for faster HPO")
    p.add_argument("--timeout", type=int, default=None, help="global timeout seconds for Optuna study")
    p.add_argument("--metric", type=str, default="best_val_reward",
                   choices=["best_val_reward", "backtest_sharpe", "final_test_reward"])
    p.add_argument("--direction", type=str, default="maximize", choices=["maximize", "minimize"])
    p.add_argument("--reward-type", type=str, choices=["dsr", "sharpe", "drawdown"], 
                   default=None, help="Fix reward type (skip HPO search)")
    p.add_argument("--reward-lookback", type=int, default=None,
                   help="Fix reward lookback (skip HPO search)")
    p.add_argument("--disable-transaction-costs", action="store_true",
                   help="Disable transaction costs (set to 0)")
    p.add_argument("--transaction-cost-rate", type=float, default=None,
                   help="Fix transaction cost rate (skip HPO search)")
    p.add_argument("--disable-inflation", action="store_true",
                   help="Disable inflation penalty (set to 0)")
    p.add_argument("--inflation-rate", type=float, default=None,
                   help="Fix inflation rate (skip HPO search)")                   
    return p.parse_args()


def suggest_params(trial, asset_class: str, encoder: str):
    """Top-impact knobs only."""
    params = {
        # PPO
        "policy_lr": trial.suggest_float("policy_lr", 1e-5, 3e-3, log=True),
        "ppo_clip_ratio": trial.suggest_float("ppo_clip_ratio", 0.08, 0.30),
        "entropy_coef": trial.suggest_float("entropy_coef", 1e-5, 5e-3, log=True),

        # Width shared by policy & encoder in your factory
        "hidden_dim": trial.suggest_categorical("hidden_dim", [256, 512, 768, 1024]),

        # Reward function parameters
        "reward_type": trial.suggest_categorical("reward_type", ["dsr", "sharpe", "drawdown"]),
        "reward_lookback": trial.suggest_int("reward_lookback", 10, 50),
    }

    # Reward shaping: DSR decay eta depends on asset class
    if asset_class == "crypto":
        params["eta"] = trial.suggest_float("eta", 0.05, 0.30, log=True)
    else:
        params["eta"] = trial.suggest_float("eta", 0.005, 0.03, log=True)

    # Encoder knobs only if VAE is used
    if encoder == "vae":
        params.update({
            "vae_lr": trial.suggest_float("vae_lr", 3e-5, 3e-3, log=True),
            "vae_beta": trial.suggest_float("vae_beta", 1e-4, 2e-1, log=True),
            "latent_dim": trial.suggest_categorical("latent_dim", [32, 64, 128, 256]),
        })
    return params

def build_cfg(base_exp: ExperimentConfig, overrides: dict, base_name: str):
    """Create TrainingConfig and apply overrides without touching repo code."""
    cfg = experiment_to_training_config(base_exp)

    # Always give each trial a unique run / checkpoint name to avoid collisions
    cfg.exp_name = base_name
    cfg.force_recreate = True

    # Device & speed tweaks for HPO
    cfg.device = overrides.pop("device")
    cfg.max_episodes = overrides.pop("max_episodes")
    cfg.val_episodes = overrides.pop("val_episodes")
    cfg.test_episodes = overrides.pop("test_episodes")
    # Encourage early stopping during HPO (still uses your built-in guardrails)
    if hasattr(cfg, "min_episodes_before_stopping"):
        cfg.min_episodes_before_stopping = min(800, max(300, cfg.max_episodes // 3))
    if hasattr(cfg, "val_interval"):
        cfg.val_interval = min(getattr(cfg, "val_interval", 200), 200)

    # Apply suggested knobs
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg

def objective_factory(args, seeds: List[int]):
    """Returns an Optuna objective that averages the metric over seeds."""
    metric_key = args.metric

    def objective(trial: optuna.trial.Trial):
        # Sample high-impact knobs
        sampled = suggest_params(
            trial, 
            args.asset_class, 
            args.encoder)

        # Common, non-sampled HPO runtime tweaks
        sampled["device"] = args.device
        sampled["max_episodes"] = args.max_episodes
        sampled["val_episodes"] = args.val_episodes
        sampled["test_episodes"] = args.test_episodes

        if args.disable_transaction_costs:
            sampled["transaction_cost_rate"] = 0.0
        elif args.transaction_cost_rate is not None:
            sampled["transaction_cost_rate"] = args.transaction_cost_rate
            
        if args.disable_inflation:
            sampled["inflation_rate"] = 0.0
        elif args.inflation_rate is not None:
            sampled["inflation_rate"] = args.inflation_rate

        # Build base experiment (seed will be set per run)
        base_exp = ExperimentConfig(
            seed=0,
            asset_class=args.asset_class,
            encoder=args.encoder,
            exp_name=None,
            force_recreate=True,
        )

        trial_scores = []
        for s in seeds:
            # Unique name per seed to avoid resume/ckpt clashes
            run_name = f"{args.study_name}_t{trial.number}_s{s}_" \
                       f"{datetime.now().strftime('%H%M%S')}"
            exp = ExperimentConfig(
                seed=s,
                asset_class=args.asset_class,
                encoder=args.encoder,
                exp_name=run_name,
                force_recreate=True,
            )
            cfg = build_cfg(
                base_exp=exp,
                overrides=dict(sampled),
                base_name=run_name,
            )

            # Train and evaluate using your existing entry point
            try:
                with mlflow.start_run(nested=True):
                    summary = run_training(cfg)
            except AttributeError as e:
                logger.error(f"AttributeError in trial: {str(e)}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                # Don't mask the error - let it bubble up or return a clear failure
                return float("-inf")  # or raise e
                # if "get_window_tensor" in str(e):
                #     # fallback: return just the metrics up to validation
                #     summary = {"best_val_reward": float("-inf")}
                # else:
                #     raise


            # Choose metric (prefer fast validation metric for HPO)
            score = summary.get(metric_key, None)
            if score is None:
                # Fallback if user selected a key not in summary
                score = summary.get("best_val_reward", None)
            if score is None:
                # Last resort: treat crash as a very poor score to let TPE learn
                score = float("-inf")

            trial_scores.append(float(score))

            # Try to free GPU/CPU memory between seeds
            try:
                import torch, gc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                pass

        # Aggregate
        mean_score = sum(trial_scores) / max(1, len(trial_scores))

        # Report once at the end (we’re not modifying your training loop for step-wise pruning)
        trial.report(mean_score, step=0)
        return mean_score

    return objective

def main():
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction=args.direction,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True),
        pruner=optuna.pruners.NopPruner(),  # zero code changes: no intermediate reports
    )

    objective = objective_factory(args, seeds)
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    print("\n=== Best trial ===")
    print("Value:", study.best_trial.value)
    print("Params:", json.dumps(study.best_trial.params, indent=2))

    # Save best params to a JSON for easy reuse
    best_path = f"best_params_{args.study_name}.json"
    with open(best_path, "w") as f:
        json.dump(study.best_trial.params, f, indent=2)
    print(f"\nSaved best params → {best_path}")

if __name__ == "__main__":
    main()
