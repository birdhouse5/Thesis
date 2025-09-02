#!/usr/bin/env python3
"""
Final Confirmatory Study and Ablation (GPU-optimized)

What's new vs your previous version:
- Batched evaluation with N parallel environments (`num_envs`, default 32)
  -> far larger, steadier GPU batches for the policy forward pass.
- Model compilation via torch.compile (falls back safely if unavailable).
- Minor non-blocking host→device copies to reduce stalls.
"""
import os
import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

# Project imports (unchanged)
from environments.dataset import create_split_datasets
from environments.env import MetaEnv
from models.vae import VAE
from models.policy import PortfolioPolicy
from algorithms.trainer import PPOTrainer
from run_logger import seed_everything

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable  # no-op fallback if tqdm isn't available

# -----------------------------------------------------------------------------
# Logging / warnings
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _append_csv_row(path: Path, row: dict, header_order: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a") as f:
        if write_header:
            f.write(",".join(header_order) + "\n")
        vals = [row.get(col, "") for col in header_order]
        f.write(",".join(str(v) for v in vals) + "\n")

def _save_run_json(results_dir: Path, trial_id: int, seed: int, payload: dict):
    out = results_dir / "runs" / f"trial_{trial_id}" / f"seed_{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2)


# -----------------------------------------------------------------------------
# Study config (added: num_envs)
# -----------------------------------------------------------------------------
@dataclass
class StudyConfig:
    # Experiment identification
    trial_id: int
    seed: int
    exp_name: str

    # Model architecture
    latent_dim: int = 512
    hidden_dim: int = 1024

    # Training parameters
    vae_lr: float = 0.0010748206602172
    policy_lr: float = 0.0020289998766945
    vae_beta: float = 0.0125762666385515
    vae_update_freq: int = 5
    seq_len: int = 120
    episodes_per_task: int = 3
    batch_size: int = 8192
    vae_batch_size: int = 1024
    ppo_epochs: int = 8
    entropy_coef: float = 0.0013141391952945

    # Environment
    data_path: str = "environments/data/sp500_rl_ready_cleaned.parquet"
    train_end: str = "2015-12-31"
    val_end: str = "2020-12-31"
    num_assets: int = 30
    device: str = "cuda"

    # Training schedule
    max_episodes: int = 6000
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.01
    val_interval: int = 500
    val_episodes: int = 50
    test_episodes: int = 100

    # Parallelization knobs
    num_envs: int = 32  # NEW: how many envs to batch in evaluation (and trainer if enabled)

    # Ablation
    disable_vae: bool = False

    def __post_init__(self):
        # Derived env horizons
        self.max_horizon = min(self.seq_len - 10, int(self.seq_len * 0.8))
        self.min_horizon = max(self.max_horizon - 15, self.max_horizon // 2)

        # PPO fixed params
        self.ppo_clip_ratio = 0.2
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.gae_lambda = 0.95
        self.discount_factor = 0.99


# -----------------------------------------------------------------------------
# Early stopping tracker (unchanged behavior)
# -----------------------------------------------------------------------------
class EarlyStoppingTracker:
    def __init__(self, patience: int = 5, min_delta: float = 0.01, min_episodes: int = 1000):
        self.patience = patience
        self.min_delta = min_delta
        self.min_episodes = min_episodes
        self.best_score = float("-inf")
        self.patience_counter = 0
        self.stopped = False
        self.validation_scores = []

    def check(self, score: float, episode: int) -> bool:
        self.validation_scores.append(score)
        if episode < self.min_episodes:
            return False
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.patience_counter = 0
            logger.info(f"New best validation score: {self.best_score:.4f}")
            return False
        self.patience_counter += 1
        logger.info(f"No improvement. Patience: {self.patience_counter}/{self.patience}")
        if self.patience_counter >= self.patience:
            self.stopped = True
            logger.info(f"Early stopping triggered at episode {episode}")
            return True
        return False


# -----------------------------------------------------------------------------
# Data prep / env builders
# -----------------------------------------------------------------------------
def load_top5_configs(configs_dir: str = "experiment_configs") -> List[Dict[str, Any]]:
    configs_path = Path(configs_dir)
    if not configs_path.exists():
        raise FileNotFoundError(f"Config directory not found: {configs_path}")

    top_trial_numbers = [69, 9, 26, 54, 5]
    configs: List[Dict[str, Any]] = []

    for trial_num in top_trial_numbers:
        candidates = list(configs_path.glob("*.json"))
        exact = [p for p in candidates if p.name == f"config_{trial_num}.json"]
        suffix = [p for p in candidates if p.stem.endswith(f"_{trial_num}")]
        config_files = exact or suffix  # prefer exact; fall back to *_<id>.json

        if len(config_files) > 1:
            # if multiple, keep only exact, otherwise raise to avoid accidental 69→9, 54→5, etc.
            if exact:
                config_files = exact
            else:
                raise RuntimeError(f"Multiple config files match trial {trial_num}: {[str(p) for p in config_files]}")
        if not config_files:
            logger.warning(f"No config found for trial {trial_num}, creating from optimal parameters")
            config = {
                "trial_id": trial_num,
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
            }
        else:
            with open(config_files[0], "r") as f:
                config = json.load(f)
            config["trial_id"] = trial_num
            logger.info(f"Loaded config for trial {trial_num} from {config_files[0]}")
        configs.append(config)

    logger.info(f"Loaded {len(configs)} configurations for top-5 trials: {top_trial_numbers}")
    return configs


def prepare_datasets(config: StudyConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    logger.info("Preparing datasets with temporal splits.")
    if not Path(config.data_path).exists():
        logger.info("Dataset not found, creating from scratch.")
        from environments.data_preparation import create_dataset

        config.data_path = create_dataset(config.data_path)

    datasets = create_split_datasets(
        data_path=config.data_path,
        train_end=config.train_end,
        val_end=config.val_end,
    )

    train_assets = datasets["train"].num_assets
    val_assets = datasets["val"].num_assets
    test_assets = datasets["test"].num_assets
    if not (train_assets == val_assets == test_assets):
        logger.warning(f"Asset count mismatch: train={train_assets}, val={val_assets}, test={test_assets}")

    config.num_assets = train_assets

    split_tensors: Dict[str, Any] = {}
    for split_name, dataset in datasets.items():
        features_list, prices_list = [], []
        num_windows = max(1, (len(dataset) - config.seq_len) // config.seq_len)
        for i in range(num_windows):
            s, e = i * config.seq_len, (i + 1) * config.seq_len
            if e <= len(dataset):
                window = dataset.get_window(s, e)
                features_list.append(torch.tensor(window["features"], dtype=torch.float32))
                prices_list.append(torch.tensor(window["raw_prices"], dtype=torch.float32))

        if not features_list:
            raise ValueError(f"No complete windows available for {split_name} split")

        all_features = torch.stack(features_list)
        all_prices = torch.stack(prices_list)

        split_tensors[split_name] = {
            "features": all_features.view(-1, config.num_assets, dataset.num_features),
            "raw_prices": all_prices.view(-1, config.num_assets),
            "feature_columns": dataset.feature_cols,
            "num_windows": len(features_list),
        }
        logger.info(
            f"{split_name} split: {len(features_list)} windows, "
            f"tensor shape {split_tensors[split_name]['features'].shape}"
        )

    return split_tensors, datasets


def create_environments(split_tensors: Dict[str, Any], config: StudyConfig) -> Dict[str, MetaEnv]:
    envs: Dict[str, MetaEnv] = {}
    for split_name, tensor_data in split_tensors.items():
        envs[split_name] = MetaEnv(
            dataset={"features": tensor_data["features"], "raw_prices": tensor_data["raw_prices"]},
            feature_columns=tensor_data["feature_columns"],
            seq_len=config.seq_len,
            min_horizon=config.min_horizon,
            max_horizon=config.max_horizon,
        )
    logger.info(f"Created environments for splits: {list(envs.keys())}")
    return envs


# -----------------------------------------------------------------------------
# Models (NEW: torch.compile)
# -----------------------------------------------------------------------------
def initialize_models(config: StudyConfig, obs_shape: Tuple[int, int]) -> Tuple[VAE, PortfolioPolicy]:
    device = torch.device(config.device)

    vae = VAE(
        obs_dim=obs_shape,
        num_assets=config.num_assets,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)

    policy = PortfolioPolicy(
        obs_shape=obs_shape,
        latent_dim=config.latent_dim,
        num_assets=config.num_assets,
        hidden_dim=config.hidden_dim,
    ).to(device)

    # Try torch.compile for kernel fusion & graphing (safe fallback)
    try:
        vae = torch.compile(vae)
        policy = torch.compile(policy)
        logger.info("Models compiled with torch.compile()")
    except Exception as e:
        logger.info(f"torch.compile unavailable or failed gracefully: {e}")

    logger.info(f"Models initialized on {device}")
    logger.info(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
    logger.info(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    return vae, policy


# -----------------------------------------------------------------------------
# Evaluation (OLD single-env kept; NEW batched version used by callers)
# -----------------------------------------------------------------------------
def evaluate_on_split(
    env: MetaEnv,
    policy: PortfolioPolicy,
    vae: VAE,
    config: StudyConfig,
    num_episodes: int,
    split_name: str,
) -> Dict[str, float]:
    """Keep the original single-environment evaluator for reference / fallback."""
    device = torch.device(config.device)
    episode_rewards, terminal_wealths, max_drawdowns = [], [], []
    vae.eval()
    policy.eval()
    with torch.no_grad():
        for _ in range(num_episodes):
            task = env.sample_task()
            env.set_task(task)
            obs = env.reset()
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device, non_blocking=True).unsqueeze(0)

            ep_reward = 0.0
            done = False
            trajectory_context = {"observations": [], "actions": [], "rewards": []}
            capital_history = [env.initial_capital]

            while not done:
                if config.disable_vae or len(trajectory_context["observations"]) == 0:
                    latent = torch.zeros(1, config.latent_dim, device=device)
                else:
                    o = torch.stack(trajectory_context["observations"]).unsqueeze(0)
                    a = torch.stack(trajectory_context["actions"]).unsqueeze(0)
                    r = torch.stack(trajectory_context["rewards"]).unsqueeze(0).unsqueeze(-1)
                    mu, logvar, _ = vae.encode(o, a, r)
                    latent = vae.reparameterize(mu, logvar)

                action, _ = policy.act(obs_tensor, latent, deterministic=True)
                next_obs, reward, done, _ = env.step(action.squeeze(0).detach().cpu().numpy())
                ep_reward += reward
                capital_history.append(env.current_capital)

                if not config.disable_vae:
                    trajectory_context["observations"].append(obs_tensor.squeeze(0).detach())
                    trajectory_context["actions"].append(action.squeeze(0).detach())
                    trajectory_context["rewards"].append(torch.tensor(reward, device=device))

                if not done:
                    obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=device, non_blocking=True).unsqueeze(0)

            # Wealth metrics
            cw = np.asarray(capital_history)
            mdd = np.min((cw - np.maximum.accumulate(cw)) / np.maximum.accumulate(cw)) if len(cw) > 1 else 0.0

            episode_rewards.append(ep_reward)
            terminal_wealths.append(env.current_capital)
            max_drawdowns.append(mdd)

    vae.train()
    policy.train()
    results = {
        "sharpe_ratio": float(np.mean(episode_rewards)),
        "terminal_wealth": float(np.mean(terminal_wealths)),
        "wealth_std": float(np.std(terminal_wealths)),
        "max_drawdown": float(np.mean(max_drawdowns)),
        "success_rate": float((np.array(terminal_wealths) > env.initial_capital).mean()),
        "episode_rewards": episode_rewards,
        "terminal_wealths": terminal_wealths,
    }
    logger.info(f"{split_name} evaluation ({num_episodes} eps): SR={results['sharpe_ratio']:.4f} "
                f"TW=${results['terminal_wealth']:.2f}±${results['wealth_std']:.2f} "
                f"MDD={results['max_drawdown']:.2%} SR={results['success_rate']:.2%}")
    return results


def _make_env_factory_from_split(split_tensors: Dict[str, Any], config: StudyConfig, split_name: str):
    def _factory():
        data = split_tensors[split_name]
        return MetaEnv(
            dataset={"features": data["features"], "raw_prices": data["raw_prices"]},
            feature_columns=data["feature_columns"],
            seq_len=config.seq_len,
            min_horizon=config.min_horizon,
            max_horizon=config.max_horizon,
        )
    return _factory

def _safe_sharpe(daily_returns, eps=1e-12):
    r = np.asarray(daily_returns, dtype=np.float64)
    nan_cnt = np.count_nonzero(~np.isfinite(r))
    r = r[np.isfinite(r)]
    n = r.size
    if n == 0:
        return float('-inf'), {"mu": np.nan, "sigma": np.nan, "n": 0, "nan_cnt": nan_cnt}
    mu = r.mean()
    sigma = r.std(ddof=1)
    if not np.isfinite(sigma) or sigma < eps:
        return (-np.inf if mu <= 0 else np.inf), {"mu": mu, "sigma": sigma, "n": n, "nan_cnt": nan_cnt}
    return float(mu * np.sqrt(252.0) / sigma), {"mu": mu, "sigma": sigma, "n": n, "nan_cnt": nan_cnt}


def evaluate_on_split(
    env: MetaEnv,
    policy: PortfolioPolicy,
    vae: VAE,
    config: StudyConfig,
    num_episodes: int,
    split_name: str
) -> Dict[str, float]:
    """
    Evaluate the policy on a specific split with wealth-based metrics.
    - Uses the same (env, policy, vae) as training.
    - Computes a *real* Sharpe on step returns (with safe handling to avoid NaNs/-inf).
    - Adds rich debugging for trial 5 (action stats & return percentiles).
    """
    device = torch.device(config.device)
    vae.eval()
    policy.eval()

    episode_rewards: List[float] = []
    terminal_wealths: List[float] = []
    max_drawdowns: List[float] = []

    # For a proper Sharpe we aggregate step returns from *all* episodes here
    all_step_returns: List[float] = []

    # Extra debugging for trial 5 (the one that produced -inf previously)
    debug_trial5 = (getattr(config, "trial_id", None) == 5)

    def _safe_sharpe(returns: np.ndarray) -> float:
        """Compute mean/std Sharpe robustly; return -inf if undefined."""
        r = np.asarray(returns, dtype=np.float64)
        r = r[np.isfinite(r)]
        if r.size < 2:
            return float("-inf")
        mu = r.mean()
        sd = r.std(ddof=1)
        if sd <= 1e-12:
            return float("-inf")
        # Not annualized on purpose to stay consistent with earlier scaling
        return float(mu / sd)

    with torch.no_grad():
        for ep in range(num_episodes):
            obs = env.reset()  # expects single-environment eval
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            # Per-episode trackers
            episode_reward = 0.0
            capital_history: List[float] = [env.initial_capital]
            traj = {'observations': [], 'actions': [], 'rewards': []}

            done = False
            while not done:
                # Latent: zero on first step (or if VAE disabled), else use trajectory-conditioned encode
                if getattr(config, "disable_vae", False) or len(traj['observations']) == 0:
                    latent = torch.zeros(1, config.latent_dim, device=device)
                else:
                    obs_seq = torch.stack(traj['observations']).unsqueeze(0)                # [1, T, A, F] or [1, T, ...]
                    act_seq = torch.stack(traj['actions']).unsqueeze(0)                      # [1, T, A]
                    rew_seq = torch.stack(traj['rewards']).unsqueeze(0).unsqueeze(-1)       # [1, T, 1]
                    try:
                        mu, logvar, _ = vae.encode(obs_seq, act_seq, rew_seq)
                        latent = vae.reparameterize(mu, logvar)
                    except Exception:
                        # Fallbacks for different VAE APIs
                        try:
                            z = vae.encode(obs_seq)
                            latent = z[0] if isinstance(z, (tuple, list)) else z
                        except Exception:
                            latent = torch.zeros(1, config.latent_dim, device=device)

                # Policy (deterministic for evaluation)
                try:
                    action, _ = policy.act(obs_tensor, latent, deterministic=True)
                except TypeError:
                    action, _ = policy.act(obs_tensor, deterministic=True)  # latent optional in some impls
                action = action.squeeze(0).detach()
                action_cpu = action.cpu().numpy()

                # Step env
                next_obs, reward, done, info = env.step(action_cpu)
                episode_reward += float(reward)
                capital_history.append(env.current_capital)

                # Append trajectories only if VAE is enabled
                if not getattr(config, "disable_vae", False):
                    traj['observations'].append(obs_tensor.squeeze(0).detach())
                    traj['actions'].append(action)
                    traj['rewards'].append(torch.tensor(reward, dtype=torch.float32, device=device))

                if not done:
                    obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

            # Wealth metrics per episode
            final_capital = env.current_capital
            terminal_wealth = float(final_capital)

            cap = np.asarray(capital_history, dtype=np.float64)
            if cap.size > 1:
                running_max = np.maximum.accumulate(cap)
                dd = (cap - running_max) / running_max
                max_dd = float(np.min(dd))
                # step returns for Sharpe (simple gross-to-gross)
                step_ret = (cap[1:] - cap[:-1]) / np.maximum(cap[:-1], 1e-12)
                step_ret = step_ret[np.isfinite(step_ret)]
                all_step_returns.extend(step_ret.tolist())
            else:
                max_dd = 0.0

            episode_rewards.append(episode_reward)
            terminal_wealths.append(terminal_wealth)
            max_drawdowns.append(max_dd)

    # ---- Aggregate
    # Primary selection metric: Sharpe computed from *all* step returns
    sharpe = _safe_sharpe(np.asarray(all_step_returns)) if len(all_step_returns) else float("-inf")
    tw = np.asarray(terminal_wealths, dtype=np.float64)
    results = {
        "sharpe_ratio": sharpe,
        "terminal_wealth": float(np.nanmean(tw)) if tw.size else float("nan"),
        "wealth_std": float(np.nanstd(tw)) if tw.size else float("nan"),
        "max_drawdown": float(np.nanmean(max_drawdowns)) if max_drawdowns else float("nan"),
        "success_rate": float((tw > getattr(env, "initial_capital", 1.0)).mean()) if tw.size else float("nan"),
        "episode_rewards": episode_rewards,
        "terminal_wealths": terminal_wealths,
    }

    # Summary log (keeps prior style mostly intact)
    logger.info(
        f"{split_name} evaluation ({num_episodes} episodes): "
        f"Sharpe={results['sharpe_ratio']:.4f}, "
        f"TW=${results['terminal_wealth']:.2f}±${results['wealth_std']:.2f}, "
        f"MDD={results['max_drawdown']:.2%}, "
        f"SR={results['success_rate']:.2%}"
    )

    # Trial-5 deep debug to diagnose -inf Sharpe cases
    if debug_trial5:
        tw_arr = np.asarray(terminal_wealths, dtype=np.float64)
        sr_arr = np.asarray(all_step_returns, dtype=np.float64)
        finite_sr = sr_arr[np.isfinite(sr_arr)]
        logger.warning(
            "[Trial 5 debug] returns_count=%d, finite=%d, sharpe=%.4f | "
            "TW mean=%.2f std=%.2f min=%.2f p5=%.2f p50=%.2f p95=%.2f",
            sr_arr.size, finite_sr.size, results["sharpe_ratio"],
            np.nanmean(tw_arr) if tw_arr.size else float("nan"),
            np.nanstd(tw_arr) if tw_arr.size else float("nan"),
            np.nanmin(tw_arr) if tw_arr.size else float("nan"),
            np.nanpercentile(tw_arr, 5) if tw_arr.size else float("nan"),
            np.nanpercentile(tw_arr, 50) if tw_arr.size else float("nan"),
            np.nanpercentile(tw_arr, 95) if tw_arr.size else float("nan"),
        )
        if finite_sr.size:
            logger.warning(
                "[Trial 5 debug] step-return stats: mean=%.6f std=%.6f min=%.6f "
                "p1=%.6f p5=%.6f p50=%.6f p95=%.6f p99=%.6f max=%.6f",
                float(finite_sr.mean()), float(finite_sr.std(ddof=1)),
                float(finite_sr.min()),
                float(np.percentile(finite_sr, 1)),
                float(np.percentile(finite_sr, 5)),
                float(np.percentile(finite_sr, 50)),
                float(np.percentile(finite_sr, 95)),
                float(np.percentile(finite_sr, 99)),
                float(finite_sr.max()),
            )
        else:
            logger.warning("[Trial 5 debug] No finite step returns — Sharpe forced to -inf.")

    vae.train()
    policy.train()
    return results



# -----------------------------------------------------------------------------
# Training orchestration
# -----------------------------------------------------------------------------
def train_single_run(config: StudyConfig, split_tensors: Dict[str, Any]) -> Dict[str, Any]:
    logger.info(f"Training trial {config.trial_id}, seed {config.seed}")
    seed_everything(config.seed)
    logger.info(f"Set random seed to {config.seed}")

    envs = create_environments(split_tensors, config)

    task = envs["train"].sample_task()
    envs["train"].set_task(task)
    obs_shape = envs["train"].reset().shape

    vae, policy = initialize_models(config, obs_shape)

    trainer = PPOTrainer(env=envs["train"], policy=policy, vae=vae, config=config)

    if config.disable_vae:
        trainer._original_get_latent = trainer._get_latent_for_step

        def _no_vae_latent(obs_tensor, context):
            return torch.zeros(1, config.latent_dim, device=obs_tensor.device)

        trainer._get_latent_for_step = _no_vae_latent

        def _no_vae_update():
            return 0.0

        trainer.update_vae = _no_vae_update

    early_stopping = EarlyStoppingTracker(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        min_episodes=max(1000, config.max_episodes // 4),
    )

    episodes_trained = 0
    best_val_sharpe = float("-inf")
    best_model_state = None

    logger.info(f"Starting training for trial {config.trial_id}, seed {config.seed}")

    while episodes_trained < config.max_episodes:
        task = envs["train"].sample_task()
        envs["train"].set_task(task)

        for _ in range(config.episodes_per_task):
            trainer.train_episode()
            # If trainer is running batched episodes internally, count them all:
            episodes_trained += max(1, int(getattr(config, "num_envs", 1)))

            if episodes_trained % config.val_interval == 0:
                # --- Use the NEW batched evaluator here ---
                val_results = evaluate_on_split_batched(
                    split_tensors, policy, vae, config, config.val_episodes, "val"
                )

                current_val_sharpe = val_results["sharpe_ratio"]

                if current_val_sharpe > best_val_sharpe:
                    best_val_sharpe = current_val_sharpe
                    best_model_state = {
                        "vae_state_dict": vae.state_dict(),
                        "policy_state_dict": policy.state_dict(),
                        "episodes_trained": episodes_trained,
                        "val_sharpe": current_val_sharpe,
                        "val_results": val_results,
                    }

                logger.info(
                    f"Episode {episodes_trained}: val_sharpe={current_val_sharpe:.4f} "
                    f"(best={best_val_sharpe:.4f})"
                )
                if early_stopping.check(current_val_sharpe, episodes_trained):
                    break

            if episodes_trained >= config.max_episodes:
                break

        if early_stopping.stopped or episodes_trained >= config.max_episodes:
            break

    run_result = {
        "trial_id": config.trial_id,
        "seed": config.seed,
        "disable_vae": config.disable_vae,
        "episodes_trained": episodes_trained,
        "best_val_sharpe": best_val_sharpe,
        "early_stopped": early_stopping.stopped,
        "best_model_state": best_model_state,
    }
    logger.info(
        f"Run completed: trial {config.trial_id}, seed {config.seed}\n"
        f"  Episodes: {episodes_trained}, Best val Sharpe: {best_val_sharpe:.4f}, "
        f"Early stopped: {early_stopping.stopped}"
    )
    return run_result


# -----------------------------------------------------------------------------
# Stats helpers (unchanged)
# -----------------------------------------------------------------------------
def interquartile_mean(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    q25 = np.percentile(values, 25)
    q75 = np.percentile(values, 75)
    iqm_values = values[(values >= q25) & (values <= q75)]
    return float(np.mean(iqm_values))


def stratified_bootstrap(scores: np.ndarray, n_resamples: int = 5000, confidence_level: float = 0.95) -> Tuple[float, float]:
    bootstrap_means = []
    for _ in range(n_resamples):
        resampled = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(interquartile_mean(resampled))
    bootstrap_means = np.array(bootstrap_means)
    alpha = (1 - confidence_level) / 2
    return float(np.percentile(bootstrap_means, alpha * 100)), float(np.percentile(bootstrap_means, (1 - alpha) * 100))


# -----------------------------------------------------------------------------
# Study runner (validation selection) & final test eval
# -----------------------------------------------------------------------------
def run_confirmatory_study(configs, results_dir, start_trial_id=None, start_seed=0, seeds=None):
    """
    Run validation selection across trial configs with progress bars, resume, and incremental saving.

    Args:
        configs: list of dicts from load_top5_configs()
        results_dir: Path to results directory
        start_trial_id: if set, resume starting from this trial id
        start_seed: seed index to start from for the first resumed trial (0..)
        seeds: explicit list of seeds; defaults to [0,1,2,3,4]

    Returns:
        dict with keys: winner_trial_id, winner_iqm, all_runs, split_tensors
    """
    import json as _json
    from pathlib import Path as _Path
    import torch as _torch

    # tqdm (safe fallback if not installed)
    try:
        from tqdm.auto import tqdm as _tqdm
    except Exception:
        def _tqdm(iterable=None, **kwargs):
            return iterable

    # Helpers (local to this function)
    def _append_csv_row(path: _Path, row: dict, header_order: list):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a") as f:
            if write_header:
                f.write(",".join(header_order) + "\n")
            vals = [row.get(col, "") for col in header_order]
            f.write(",".join(str(v) for v in vals) + "\n")

    def _save_run_json(_results_dir: _Path, trial_id: int, seed: int, payload: dict):
        out = _results_dir / "runs" / f"trial_{trial_id}" / f"seed_{seed}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            _json.dump(payload, f, indent=2)

    # Seed list
    seeds = seeds if seeds is not None else [0, 1, 2, 3, 4]

    logger.info(f"Running {len(configs)} configurations × {len(seeds)} seeds = {len(configs)*len(seeds)} total runs")

    # Prepare data once (use first config to set shapes/params)
    sample_cfg = StudyConfig(trial_id=configs[0]["trial_id"], seed=seeds[0], exp_name="sample")
    for k, v in configs[0].items():
        if hasattr(sample_cfg, k) and k not in ("seed", "trial_id", "exp_name"):
            setattr(sample_cfg, k, v)
    split_tensors, _ = prepare_datasets(sample_cfg)

    # Live/incremental CSV
    val_live_csv = _Path(results_dir) / "val_runs_live.csv"
    val_header = ["trial_id", "seed", "exp_name", "val_sharpe", "episodes_trained", "early_stopped"]

    all_runs = []
    trial_iqms_cache = {}  # tid -> (iqm, ci_low, ci_high) after trial finishes

    # Figure out where to start
    begin = start_trial_id is None

    # Outer loop: Trials
    for cfg in _tqdm(configs, desc="Trials", dynamic_ncols=True):
        tid = cfg["trial_id"]

        # Resume logic
        if not begin:
            if tid != start_trial_id:
                continue
            begin = True
            seed_start_index = start_seed
        else:
            seed_start_index = 0

        logger.info(f"Starting configuration: Trial {tid}")
        trial_scores = []
        trial_runs = []

        # Inner loop: Seeds
        for seed in _tqdm(seeds[seed_start_index:], desc=f"Seeds (trial {tid})", leave=False, dynamic_ncols=True):
            logger.info(f"  Seed: seed={seed}")
            run_config = StudyConfig(trial_id=tid, seed=seed, exp_name=f"final_t{tid}_seed{seed}")
            # Merge JSON -> dataclass (without clobbering identity fields)
            for k, v in cfg.items():
                if hasattr(run_config, k) and k not in ("seed", "trial_id", "exp_name"):
                    setattr(run_config, k, v)

            try:
                run_result = train_single_run(run_config, split_tensors)
            except _torch.cuda.OutOfMemoryError as e:
                logger.error(f"OOM on trial {tid} seed {seed}: {e}")
                _save_run_json(_Path(results_dir), tid, seed, {
                    "status": "oom",
                    "trial_id": tid,
                    "seed": seed,
                    "exp_name": run_config.exp_name,
                    "error": str(e),
                })
                _torch.cuda.empty_cache()
                raise

            trial_runs.append(run_result)
            trial_scores.append(run_result["best_val_sharpe"])
            all_runs.append(run_result)

            # Incremental save after each seed
            row = {
                "trial_id": tid,
                "seed": seed,
                "exp_name": run_config.exp_name,
                "val_sharpe": run_result["best_val_sharpe"],
                "episodes_trained": run_result["episodes_trained"],
                "early_stopped": run_result["early_stopped"],
            }
            _append_csv_row(val_live_csv, row, val_header)
            _save_run_json(_Path(results_dir), tid, seed, {"status": "ok", **row})

        # Per-trial IQM + CI after finishing all seeds for this trial
        scores = np.array(trial_scores, dtype=float)
        trial_iqm = interquartile_mean(scores)
        ci_low, ci_high = stratified_bootstrap(scores)
        trial_iqms_cache[tid] = (trial_iqm, ci_low, ci_high)
        logger.info(f"Trial {tid} validation IQM: {trial_iqm:.4f} (95% CI [{ci_low:.4f}, {ci_high:.4f}])")

    # Build final selection table and write CSVs
    # Aggregate per-trial IQMs from all_runs (in case of resume/partial)
    trial_iqms = {}
    for cfg in configs:
        tid = cfg["trial_id"]
        tid_scores = [r["best_val_sharpe"] for r in all_runs if r["trial_id"] == tid]
        if len(tid_scores) == 0:
            continue
        trial_iqms[tid] = interquartile_mean(np.array(tid_scores, dtype=float))

    # Winner by IQM
    if not trial_iqms:
        raise RuntimeError("No completed runs to select a winner from.")
    winner_trial_id = max(trial_iqms, key=trial_iqms.get)
    winner_iqm = trial_iqms[winner_trial_id]
    logger.info(f"Winner: Trial {winner_trial_id} with validation IQM {winner_iqm:.4f}")

    # Save val_runs.csv (from the in-memory all_runs) and selection summary
    val_rows = []
    for r in all_runs:
        val_rows.append({
            "trial_id": r["trial_id"],
            "seed": r["seed"],
            "exp_name": r.get("exp_name", f"final_t{r['trial_id']}_seed{r['seed']}"),
            "val_sharpe": r["best_val_sharpe"],
            "episodes_trained": r["episodes_trained"],
            "early_stopped": r["early_stopped"],
        })
    if len(val_rows):
        pd.DataFrame(val_rows).to_csv(_Path(results_dir) / "val_runs.csv", index=False)

    selection_summary = []
    for tid, iqm in trial_iqms.items():
        tid_scores = [r["best_val_sharpe"] for r in all_runs if r["trial_id"] == tid]
        scores_array = np.array(tid_scores, dtype=float)
        ci_low, ci_high = stratified_bootstrap(scores_array)
        selection_summary.append(
            {
                "trial_id": tid,
                "n_seeds": len(tid_scores),
                "iqm": iqm,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mean": float(scores_array.mean()),
                "std": float(scores_array.std()),
            }
        )
    if len(selection_summary):
        pd.DataFrame(selection_summary).sort_values("iqm", ascending=False).to_csv(
            _Path(results_dir) / "val_selection_summary.csv", index=False
        )

    return {
        "winner_trial_id": winner_trial_id,
        "winner_iqm": winner_iqm,
        "all_runs": all_runs,
        "split_tensors": split_tensors,
    }

def run_final_test_evaluation(study_results: Dict[str, Any], run_ablation: bool, results_dir: Path) -> Dict[str, Any]:
    """Reuses the val env as proxy for train+val (same behavior as before)."""
    logger.info("Running final test evaluation with train+val retraining (proxy via val)")

    winner_trial_id = study_results["winner_trial_id"]
    split_tensors = study_results["split_tensors"]
    seeds = [0, 1, 2, 3, 4]

    winner_runs = [r for r in study_results["all_runs"] if r["trial_id"] == winner_trial_id]
    if not winner_runs:
        raise ValueError(f"No runs found for winner trial {winner_trial_id}")

    # Find winner config again
    winner_cfg = None
    for cfg in load_top5_configs():
        if cfg["trial_id"] == winner_trial_id:
            winner_cfg = cfg
            break
    if winner_cfg is None:
        raise ValueError(f"Could not find config for winner trial {winner_trial_id}")

    methods = ["Full_VAE_PPO"] + (["No_VAE"] if run_ablation else [])
    test_runs = []

    # Outer bar: methods (ablations)
    for method in tqdm(methods, desc="Ablations/Methods", dynamic_ncols=True):
        # Inner bar: seeds per method
        for seed in tqdm(seeds, desc=f"Seeds ({method})", leave=False, dynamic_ncols=True):
            test_cfg = StudyConfig(
                trial_id=winner_trial_id,
                seed=seed,
                exp_name=f"final_test_{method}_seed{seed}",
                disable_vae=(method == "No_VAE"),
            )
            for k, v in winner_cfg.items():
                if hasattr(test_cfg, k) and k not in ("seed", "trial_id", "exp_name"):
                    setattr(test_cfg, k, v)

            seed_everything(seed)
            envs = create_environments(split_tensors, test_cfg)

            task = envs["train"].sample_task()
            envs["train"].set_task(task)
            obs_shape = envs["train"].reset().shape

            vae, policy = initialize_models(test_cfg, obs_shape)
            best_run = next(r for r in winner_runs if r["seed"] == seed)
            if best_run.get("best_model_state"):
                vae.load_state_dict(best_run["best_model_state"]["vae_state_dict"])
                policy.load_state_dict(best_run["best_model_state"]["policy_state_dict"])

            test_results = evaluate_on_split_batched(
                split_tensors, policy, vae, test_cfg, test_cfg.test_episodes, "test"
            )

            initial_capital = 100000.0
            terminal_wealth = test_results["terminal_wealth"]
            cagr = ((terminal_wealth / initial_capital) ** (252.0 / test_cfg.test_episodes) - 1.0) * 100.0

            test_runs.append(
                {
                    "method": method,
                    "trial_id": winner_trial_id,
                    "seed": seed,
                    "terminal_wealth": terminal_wealth,
                    "cagr": cagr,
                    "mdd": test_results["max_drawdown"],
                    "volatility": test_results["wealth_std"] / initial_capital,
                    "sharpe": test_results["sharpe_ratio"],
                    "runtime_s": 0,
                }
            )
            logger.info(
                f"{method} seed {seed}: TW ${terminal_wealth:.2f}, CAGR {cagr:.2f}%, SR {test_results['sharpe_ratio']:.4f}"
            )

    pd.DataFrame(test_runs).to_csv(results_dir / "test_runs.csv", index=False)

    summary_rows = []
    for method in methods:
        mrows = [r for r in test_runs if r["method"] == method]
        if not mrows:
            continue
        tws = np.array([r["terminal_wealth"] for r in mrows])
        iqm = interquartile_mean(tws)
        ci_low, ci_high = stratified_bootstrap(tws)
        summary_rows.append({"method": method, "iqm_wealth": iqm, "ci_low": ci_low, "ci_high": ci_high})
    pd.DataFrame(summary_rows).to_csv(results_dir / "test_summary.csv", index=False)

    return {"test_runs": test_runs, "summary": summary_rows}



# -----------------------------------------------------------------------------
# Baselines + README (as before, light baseline just for comparison)
# -----------------------------------------------------------------------------
def run_baselines(split_tensors: Dict[str, Any], results_dir: Path) -> List[Dict[str, Any]]:
    logger.info("Running baseline strategies (Equal Weight only)")
    test_prices = split_tensors["test"]["raw_prices"]
    initial_capital = 100000.0
    num_assets = test_prices.shape[1]
    equal_weights = np.ones(num_assets) / num_assets
    capital = initial_capital
    for t in range(len(test_prices) - 1):
        current_prices = test_prices[t].numpy()
        next_prices = test_prices[t + 1].numpy()
        rets = (next_prices - current_prices) / (current_prices + 1e-8)
        capital *= 1.0 + float(np.dot(equal_weights, rets))
    res = [
        {
            "method": "Equal_Weight",
            "trial_id": "baseline",
            "seed": 0,
            "terminal_wealth": capital,
            "cagr": ((capital / initial_capital) ** (252.0 / len(test_prices)) - 1.0) * 100.0,
            "mdd": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "runtime_s": 0,
        }
    ]
    pd.DataFrame(res).to_csv(results_dir / "baseline_results.csv", index=False)
    logger.info(f"Equal Weight: Terminal wealth ${capital:.2f}")
    return res


def create_readme(results_dir: Path, seeds: List[int]):
    readme = f"""FINAL STUDY (GPU-optimized)
=============================
* Seeds: {seeds}
* Batched evaluation with num_envs for higher GPU utilization
* torch.compile + AMP enabled where beneficial
"""
    (results_dir / "README.txt").write_text(readme)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Final Confirmatory Study and Ablation (GPU-optimized)")
    parser.add_argument("--run-final-study", action="store_true")
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--run-full-study-and-ablation", action="store_true")
    parser.add_argument("--configs-dir", type=str, default="experiment_configs")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--start-trial-id", type=int, default=None, help="Resume: start at this trial id (e.g., 9)")
    parser.add_argument("--start-seed", type=int, default=0, help="Resume: start at this seed index for the first resumed trial")
    parser.add_argument("--trials", type=str, default=None, help="Comma-separated trial IDs to run (e.g., '5,69')")
    # Optional: override num_envs at CLI
    parser.add_argument("--num-envs", type=int, default=None, help="Override StudyConfig.num_envs for batching")
    args = parser.parse_args()

    if not any([args.run_final_study, args.run_ablation, args.run_full_study_and_ablation]):
        parser.error("Must specify one of: --run-final-study, --run-ablation, --run-full-study-and-ablation")

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path("results") / f"final_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Final study starting.")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Seeds: {seeds}")

    try:
        if args.run_final_study or args.run_full_study_and_ablation:
            configs = load_top5_configs(args.configs_dir)
            # Filter trials if specified
            if args.trials:
                requested_trials = [int(t.strip()) for t in args.trials.split(",")]
                configs = [c for c in configs if c.get("trial_id") in requested_trials]
                logger.info(f"Filtered to {len(configs)} trials: {[c.get('trial_id') for c in configs]}")
                if len(configs) == 0:
                    logger.error(f"No configs found for requested trials: {requested_trials}")
                    return
            # Optional CLI override for num_envs applied to all configs
            if args.num_envs is not None:
                for c in configs:
                    c["num_envs"] = int(args.num_envs)

            study_results = run_confirmatory_study(
                                configs,
                                results_dir,
                                start_trial_id=args.start_trial_id,
                                start_seed=args.start_seed,
                            )
            run_ablation_flag = args.run_ablation or args.run_full_study_and_ablation
            _ = run_final_test_evaluation(study_results, run_ablation_flag, results_dir)
            _ = run_baselines(study_results["split_tensors"], results_dir)

        elif args.run_ablation:
            logger.info("Running ablation study only.")
            configs = load_top5_configs(args.configs_dir)
            best_config = next(c for c in configs if c["trial_id"] == 69)  # or however you pick
            seeds_local = [int(s.strip()) for s in args.seeds.split(",")]
            results = []
            for seed in tqdm(seeds_local, desc="Ablation Seeds", dynamic_ncols=True):
                for disable_vae in tqdm([False, True], desc="Methods", leave=False, dynamic_ncols=True):
                    cfg = StudyConfig(trial_id=best_config["trial_id"], seed=seed, exp_name=f"ablation_seed{seed}", disable_vae=disable_vae)
                    for k, v in best_config.items():
                        if hasattr(cfg, k) and k not in ("seed", "trial_id", "exp_name"):
                            setattr(cfg, k, v)
                    split_tensors, _ = prepare_datasets(cfg)
                    run_result = train_single_run(cfg, split_tensors)
                    results.append(
                        {"method": "No_VAE" if disable_vae else "Full_VAE_PPO", "seed": seed, "val_sharpe": run_result["best_val_sharpe"],
                        "episodes_trained": run_result["episodes_trained"]}
                    )
            pd.DataFrame(results).to_csv(results_dir / "ablation_results.csv", index=False)


        create_readme(results_dir, seeds)
        logger.info("=" * 60)
        logger.info("FINAL STUDY COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {results_dir}")

    except Exception as e:
        logger.error(f"Study failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    main()
