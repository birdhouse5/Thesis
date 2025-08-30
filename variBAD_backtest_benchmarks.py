#!/usr/bin/env python3
"""
variBAD_backtest_benchmarks.py
-------------------------------------------------------------------------------
Standalone script that merges:
- Model loading and discovery (from loader.py)
- Sequential backtest of trained models (from backtester.py)
- Inlined benchmark strategies + engine (from backtest_engine.py)

What it does (default behavior):
1) Loads your dataset splits (train/val/test) via environments.dataset.create_split_datasets
2) Runs backtests on all auto-discovered trained models (see --search-dirs), on the *same* test window
3) Runs a suite of benchmark strategies on that same test window
4) Saves a single day-level, long-form CSV with all models and benchmarks for easy comparison/visualization

CSV columns (long-form, day-level):
    date, wealth, returns, model_name, phase, trial_number, kind, strategy_type, num_assets, weights, cash_weight

Optionally also saves:
- A wide CSV (wealth by model/benchmark) for plotting
- Parquet versions of both

Usage examples:
    python variBAD_backtest_benchmarks.py \
        --data-path environments/data/sp500_rl_ready_cleaned.parquet \
        --train-end 2018-12-31 --val-end 2020-12-31

    # Skip models, just benchmarks
    python variBAD_backtest_benchmarks.py --no-models --run-benchmarks

    # Specific benchmark list
    python variBAD_backtest_benchmarks.py --benchmarks equal_weight momentum risk_parity_invvol

    # Search specific directories for checkpoints
    python variBAD_backtest_benchmarks.py --search-dirs results/optuna_phase1_runs validation_results

Notes:
- This script relies on your existing project modules:
    environments.dataset.create_split_datasets
    models.vae.VAE
    models.policy.PortfolioPolicy
- If those modules aren't importable in your PYTHONPATH, adjust accordingly.
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Required external modules from your repo
from environments.dataset import create_split_datasets
# We don't need MetaEnv directly here

# =============================================================================
# Logging
# =============================================================================
logger = logging.getLogger("variBAD_backtest_benchmarks")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# =============================================================================
# Model Loader (merged from loader.py)
# =============================================================================
class ModelLoader:
    """Generic loader for VariBAD model checkpoints"""
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        metadata = {
            "path": str(checkpoint_path),
            "trial_number": checkpoint.get("trial_number", "unknown"),
            "episodes_trained": checkpoint.get("episodes_trained", "unknown"),
            "best_val_sharpe": checkpoint.get("best_val_sharpe", "unknown"),
            "config": checkpoint.get("config", {}),
            "checkpoint_data": checkpoint,
        }
        return metadata

    def extract_config(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        config = metadata.get("config", {})
        defaults = {
            "latent_dim": 64,
            "hidden_dim": 256,
            "num_assets": 30,
            "seq_len": 60,
            "max_horizon": 45,
            "min_horizon": 30,
            "data_path": "environments/data/sp500_rl_ready_cleaned.parquet",
            "train_end": "2015-12-31",
            "val_end": "2020-12-31",
            "device": "cuda",
        }
        for k, v in defaults.items():
            if k not in config:
                config[k] = v
                logger.warning(f"Config missing {k}, using default: {v}")
        if "max_horizon" not in config:
            config["max_horizon"] = min(config["seq_len"] - 10, int(config["seq_len"] * 0.8))
        if "min_horizon" not in config:
            config["min_horizon"] = max(config["max_horizon"] - 15, config["max_horizon"] // 2)
        return config

    def create_models(self, config: Dict[str, Any], obs_shape: Tuple[int, int]):
        # Local import to avoid hard dependency if not needed
        from models.vae import VAE
        from models.policy import PortfolioPolicy

        vae = VAE(
            obs_dim=obs_shape,
            num_assets=config["num_assets"],
            latent_dim=config["latent_dim"],
            hidden_dim=config["hidden_dim"],
        ).to(self.device)

        policy = PortfolioPolicy(
            obs_shape=obs_shape,
            latent_dim=config["latent_dim"],
            num_assets=config["num_assets"],
            hidden_dim=config["hidden_dim"],
        ).to(self.device)
        return vae, policy

    def load_model_states(self, models, checkpoint_data: Dict[str, Any]) -> None:
        vae, policy = models
        if "vae_state_dict" in checkpoint_data:
            vae.load_state_dict(checkpoint_data["vae_state_dict"])
        else:
            logger.warning("No VAE state dict found in checkpoint")
        if "policy_state_dict" in checkpoint_data:
            policy.load_state_dict(checkpoint_data["policy_state_dict"])
        else:
            logger.warning("No Policy state dict found in checkpoint")
        vae.eval()
        policy.eval()

    def load_complete_model(self, checkpoint_path: str, obs_shape: Tuple[int, int]):
        metadata = self.load_checkpoint(checkpoint_path)
        config = self.extract_config(metadata)
        vae, policy = self.create_models(config, obs_shape)
        self.load_model_states((vae, policy), metadata["checkpoint_data"])

        # Post-process metadata (phase, trial)
        path_parts = Path(checkpoint_path).parts
        metadata["phase"] = "unknown"
        for part in path_parts:
            lower = part.lower()
            if "phase1" in lower:
                metadata["phase"] = "phase1"
            elif "phase2" in lower:
                metadata["phase"] = "phase2"
            elif "phase3" in lower:
                metadata["phase"] = "phase3"
        for part in path_parts:
            if part.startswith("trial_"):
                try:
                    metadata["trial_number"] = int(part.split("_")[1])
                    break
                except (IndexError, ValueError):
                    pass
        metadata.setdefault("model_type", "unknown")
        return vae, policy, metadata


def auto_discover_models(search_dirs: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if search_dirs is None:
        search_dirs = [
            "results/optuna_phase1_runs",
            "results/optuna_phase2_runs",
            "results/optuna_phase3_runs",
            "optuna_phase1_results",
            "optuna_phase2_results",
            "optuna_phase3_results",
            "validation_results",
        ]

    found: List[Dict[str, Any]] = []
    for search_dir in search_dirs:
        search_path = Path(search_dir)
        if not search_path.exists():
            logger.debug(f"Directory not found: {search_path}")
            continue

        phase = "unknown"
        lower = str(search_dir).lower()
        if "phase1" in lower:
            phase = "phase1"
        elif "phase2" in lower:
            phase = "phase2"
        elif "phase3" in lower:
            phase = "phase3"

        # trial_* subdirs
        for trial_dir in search_path.glob("trial_*"):
            if not trial_dir.is_dir():
                continue
            try:
                trial_num = int(trial_dir.name.split("_")[1])
            except (IndexError, ValueError):
                trial_num = "unknown"

            best_model = trial_dir / "best_model.pt"
            if best_model.exists():
                found.append({
                    "path": str(best_model),
                    "name": f"{phase}_trial_{trial_num}",
                    "phase": phase,
                    "trial_number": trial_num,
                    "source": search_path.name,
                    "model_type": "best_model",
                })

            for final_model in trial_dir.glob("final_*.pt"):
                found.append({
                    "path": str(final_model),
                    "name": f"{phase}_trial_{trial_num}_{final_model.stem}",
                    "phase": phase,
                    "trial_number": trial_num,
                    "source": search_path.name,
                    "model_type": "final_model",
                })

        # phase-level best
        for best_model in search_path.glob("best_model.pt"):
            found.append({
                "path": str(best_model),
                "name": f"{phase}_phase_best",
                "phase": phase,
                "trial_number": "phase_best",
                "source": search_path.name,
                "model_type": "phase_best",
            })

    def sort_key(m):
        phase_order = {"phase1": 1, "phase2": 2, "phase3": 3, "unknown": 4}
        phase_num = phase_order.get(m["phase"], 4)
        trial = m["trial_number"]
        tnum = trial if isinstance(trial, int) else 9999
        return (phase_num, tnum)

    found.sort(key=sort_key)
    # Summaries in log
    by_phase = {}
    for m in found:
        by_phase.setdefault(m["phase"], 0)
        by_phase[m["phase"]] += 1
    for ph, cnt in by_phase.items():
        logger.info(f"Discovered {cnt} models in {ph}")
    return found


# =============================================================================
# Benchmark Strategies (merged from backtest_engine.py)
# =============================================================================
class BaseStrategy:
    name: str = "base"
    def get_action(self, obs: np.ndarray, context: Dict) -> np.ndarray:
        raise NotImplementedError

class EqualWeightStrategy(BaseStrategy):
    name = "equal_weight"
    def __init__(self, num_assets: int):
        self.num_assets = num_assets
    def get_action(self, obs: np.ndarray, context: Dict) -> np.ndarray:
        return (np.ones(self.num_assets, dtype=np.float32) / float(self.num_assets)).astype(np.float32)

class MarketCapWeightedStrategy(BaseStrategy):
    name = "market_cap"
    def __init__(self, num_assets: int):
        self.num_assets = num_assets
    def get_action(self, obs: np.ndarray, context: Dict) -> np.ndarray:
        prices = context.get("current_prices", None)
        if prices is None or len(prices) != self.num_assets:
            prices = obs[:, 0]
        prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
        prices = np.clip(prices, 0.0, None)
        if prices.sum() <= 0:
            return np.ones(self.num_assets, dtype=np.float32) / float(self.num_assets)
        w = prices / prices.sum()
        return w.astype(np.float32)

class MomentumStrategy(BaseStrategy):
    name = "momentum"
    def __init__(self, num_assets: int, lookback: int = 60, eps: float = 1e-8):
        self.num_assets = num_assets
        self.lookback = lookback
        self.eps = eps
    def get_action(self, obs: np.ndarray, context: Dict) -> np.ndarray:
        hist = context.get("trajectory_context", {}).get("returns", [])
        if len(hist) >= self.lookback:
            window = np.stack(hist[-self.lookback:], axis=0)
            mom = window.mean(axis=0)
        else:
            mom = obs[:, 1] if obs.shape[1] > 1 else np.ones(self.num_assets)
        mom = np.nan_to_num(mom, nan=0.0, posinf=0.0, neginf=0.0)
        pos = np.clip(mom, 0.0, None)
        if pos.sum() <= self.eps:
            w = np.ones(self.num_assets, dtype=np.float32) / float(self.num_assets)
        else:
            w = pos / (pos.sum() + self.eps)
        return w.astype(np.float32)

class RiskParityInvVolStrategy(BaseStrategy):
    name = "risk_parity_invvol"
    def __init__(self, num_assets: int, lookback: int = 60, eps: float = 1e-8):
        self.num_assets = num_assets
        self.lookback = lookback
        self.eps = eps
    def get_action(self, obs: np.ndarray, context: Dict) -> np.ndarray:
        hist = context.get("trajectory_context", {}).get("returns", [])
        if len(hist) < self.lookback:
            return np.ones(self.num_assets, dtype=np.float32) / float(self.num_assets)
        window = np.stack(hist[-self.lookback:], axis=0)
        vols = window.std(axis=0) + self.eps
        inv_vol = 1.0 / vols
        inv_vol = np.clip(np.nan_to_num(inv_vol, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        if inv_vol.sum() <= self.eps:
            return np.ones(self.num_assets, dtype=np.float32) / float(self.num_assets)
        return (inv_vol / inv_vol.sum()).astype(np.float32)

class RandomLongOnlyStrategy(BaseStrategy):
    name = "random"
    def __init__(self, num_assets: int, seed: int = 42):
        self.num_assets = num_assets
        self.rng = np.random.default_rng(seed)
    def get_action(self, obs: np.ndarray, context: Dict) -> np.ndarray:
        x = self.rng.random(self.num_assets)
        if x.sum() == 0:
            return np.ones(self.num_assets, dtype=np.float32) / float(self.num_assets)
        return (x / x.sum()).astype(np.float32)

class BuyAndHoldSharesStrategy(BaseStrategy):
    name = "buy_and_hold_shares"
    def __init__(self, num_assets: int, init_weights: Optional[np.ndarray] = None):
        self.num_assets = num_assets
        self.init_weights = init_weights
        self._shares: Optional[np.ndarray] = None
    def get_action(self, obs: np.ndarray, context: Dict) -> np.ndarray:
        prices = context.get("current_prices", None)
        capital = float(context.get("current_capital", 1.0))
        if prices is None or len(prices) != self.num_assets:
            return np.ones(self.num_assets, dtype=np.float32) / float(self.num_assets)
        prices = np.clip(np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0), 1e-8, None)
        if self._shares is None:
            w0 = (
                self.init_weights
                if self.init_weights is not None
                else (np.ones(self.num_assets) / float(self.num_assets))
            ).astype(np.float32)
            w0 = np.clip(w0, 0.0, None)
            s = w0.sum()
            w0 = (w0 / s) if s > 0 else (np.ones(self.num_assets, dtype=np.float32) / float(self.num_assets))
            dollars = capital * w0
            self._shares = dollars / prices
        position_value = self._shares * prices
        tot = position_value.sum()
        if tot <= 0:
            return np.ones(self.num_assets, dtype=np.float32) / float(self.num_assets)
        return (position_value / tot).astype(np.float32)

@dataclass
class BenchmarkFactory:
    num_assets: int
    seed: int = 42
    def create_all(self, exclude: Optional[List[str]] = None) -> Dict[str, BaseStrategy]:
        exclude = set([e.lower() for e in (exclude or [])])
        all_defs = {
            "equal_weight": EqualWeightStrategy(self.num_assets),
            "market_cap": MarketCapWeightedStrategy(self.num_assets),
            "momentum": MomentumStrategy(self.num_assets),
            "risk_parity_invvol": RiskParityInvVolStrategy(self.num_assets),
            "random": RandomLongOnlyStrategy(self.num_assets, seed=self.seed),
            "buy_and_hold_shares": BuyAndHoldSharesStrategy(self.num_assets),
        }
        return {k: v for k, v in all_defs.items() if k.lower() not in exclude}

    def from_names(self, names: List[str]) -> Dict[str, BaseStrategy]:
        mapping = {
            "equal": "equal_weight",
            "equal_weight": "equal_weight",
            "ew": "equal_weight",
            "market_cap": "market_cap",
            "mktcap": "market_cap",
            "cap": "market_cap",
            "momentum": "momentum",
            "mom": "momentum",
            "risk_parity": "risk_parity_invvol",
            "invvol": "risk_parity_invvol",
            "risk_parity_invvol": "risk_parity_invvol",
            "inverse_vol": "risk_parity_invvol",
            "random": "random",
            "rnd": "random",
            "buy_and_hold": "buy_and_hold_shares",
            "bah": "buy_and_hold_shares",
            "buy_and_hold_shares": "buy_and_hold_shares",
        }
        resolved = []
        for n in names:
            key = mapping.get(n.lower().strip())
            if not key:
                raise ValueError(f"Unknown benchmark strategy: {n}")
            resolved.append(key)
        return self.create_all(exclude=[k for k in self.create_all().keys() if k not in resolved])


# =============================================================================
# Engine (unified)
# =============================================================================
class UnifiedBacktestEngine:
    def __init__(
        self,
        data_path: str,
        train_end: str,
        val_end: str,
        initial_capital: float = 100000.0,
        device: str = "cuda",
        seed: int = 42,
    ):
        self.data_path = data_path
        self.train_end = train_end
        self.val_end = val_end
        self.initial_capital = initial_capital
        self.seed = seed
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Loaded test tensors
        self.test_data: Dict[str, Any] = {}
        self.obs_shape: Optional[Tuple[int, int]] = None

        # Context cap for strategies/models
        self.max_context = 100

        # Helpers
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.model_loader = ModelLoader(device=device)

        # Results dir
        self.results_dir = Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)



        # --- PATCH START: helper to pull the full test window ---
    def _fetch_full_test_window(self, test_dataset):
        """
        Return a dict with full-span arrays:
            {"features": np.ndarray [T,N,F], "raw_prices": np.ndarray [T,N], "dates": Optional[1D]}
        Works even if len(test_dataset) is NOT the time length.
        """
        import numpy as np
        import torch

        # Case A: the dataset already exposes full arrays
        for attr in ("data", "dataset", "store"):
            if hasattr(test_dataset, attr):
                d = getattr(test_dataset, attr)
                if isinstance(d, dict):
                    feats = d.get("features")
                    prices = d.get("raw_prices") or d.get("prices")
                    dates  = d.get("date") or d.get("dates")
                    if feats is not None and prices is not None:
                        return {
                            "features": np.asarray(feats),
                            "raw_prices": np.asarray(prices),
                            "dates": np.asarray(dates) if dates is not None else None,
                        }

        # Case B: expand a get_window(0, end) until it stops growing
        if hasattr(test_dataset, "get_window"):
            prev_len = -1
            end = 1
            last = None
            # Exponential growth; stop when returned length no longer increases
            for _ in range(32):  # safety cap
                try:
                    w = test_dataset.get_window(0, end)
                    f = w["features"]
                    cur_len = f.shape[0]
                    last = w
                    if cur_len <= prev_len:
                        break
                    prev_len = cur_len
                    end *= 2
                except Exception:
                    # If we overshoot, stop and use the last good window
                    break
            if last is None:
                raise RuntimeError("get_window probing failed; cannot extract test window.")
            feats = np.asarray(last["features"])
            prices = np.asarray(last["raw_prices"])
            dates  = np.asarray(last.get("dates")) if "dates" in last else None
            return {"features": feats, "raw_prices": prices, "dates": dates}

        raise TypeError("Unsupported test dataset type: cannot extract features/raw_prices.")


    # -----------------------------
    # Data setup (robust, merged)
    # -----------------------------
    def setup_test_environment(self) -> tuple[int, int]:
        logger.info("Loading dataset and preparing splits...")
        splits = create_split_datasets(self.data_path, self.train_end, self.val_end)
        test_dataset = splits["test"]

        # Pull the *full* test span (not just len(test_dataset))
        win = self._fetch_full_test_window(test_dataset)
        features = win["features"]     # [T, N, F]
        raw_prices = win["raw_prices"] # [T, N]
        dates = win.get("dates", None)

        # Torch tensors
        features_t = torch.as_tensor(features, dtype=torch.float32)
        raw_prices_t = torch.as_tensor(raw_prices, dtype=torch.float32)

        # Dates alignment
        T_feat = features_t.shape[0]
        try:
            if dates is None:
                # Try common places for dates on the dataset object
                if hasattr(test_dataset, "data"):
                    d = getattr(test_dataset, "data")
                    if isinstance(d, dict):
                        dates = d.get("date") or d.get("dates")
                if dates is None and hasattr(test_dataset, "dates"):
                    dates = getattr(test_dataset, "dates")
        except Exception:
            dates = None

        if dates is not None:
            dates = pd.to_datetime(pd.Series(np.asarray(dates))).sort_values().to_numpy()
            if len(dates) >= T_feat:
                dates = dates[:T_feat]
            else:
                # Extend sequentially day-by-day if dates are shorter than features
                start = pd.to_datetime(dates[0]) if len(dates) > 0 else pd.Timestamp("1970-01-01")
                dates = pd.date_range(start=start, periods=T_feat, freq="D")
        else:
            logger.warning("No dates found on test dataset; constructing a synthetic daily date index.")
            dates = pd.date_range(start=pd.Timestamp("1970-01-01"), periods=T_feat, freq="D")

        self.test_data = {"features": features_t, "raw_prices": raw_prices_t, "dates": pd.to_datetime(dates)}
        self.obs_shape = (features_t.shape[1], features_t.shape[2])  # (N, F)

        # Helpful confirmation in logs
        d0 = pd.to_datetime(dates[0]).date()
        d1 = pd.to_datetime(dates[-1]).date()
        logger.info(f"Test environment ready. T={T_feat}, N={features_t.shape[1]}, F={features_t.shape[2]}, "
                    f"Date range: {d0} → {d1}")
        return self.obs_shape


    # -----------------------------
    # Utilities
    # -----------------------------
    @staticmethod
    def _normalize_weights(w: np.ndarray) -> np.ndarray:
        w = np.nan_to_num(np.asarray(w, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        w = np.clip(w, 0.0, 1.0)
        s = w.sum()
        if s > 1.0 + 1e-9:
            w = w / s
        return w.astype(np.float32)

    @staticmethod
    def _max_drawdown(wealth_series: pd.Series) -> float:
        cummax = wealth_series.cummax()
        drawdown = (wealth_series - cummax) / cummax.replace(0, np.nan)
        return float(drawdown.min()) if len(drawdown) else 0.0

    # -----------------------------
    # Model backtesting (sequential)
    # -----------------------------
    def backtest_single_model(self, model_info: Dict[str, Any]) -> pd.DataFrame:
        assert self.obs_shape is not None, "Call setup_test_environment() first"
        vae, policy, metadata = self.model_loader.load_complete_model(model_info["path"], self.obs_shape)
        metadata.update(model_info)

        features = self.test_data["features"]
        prices = self.test_data["raw_prices"]
        dates = self.test_data["dates"]

        current_capital = float(self.initial_capital)
        results: List[Dict[str, Any]] = []

        trajectory: Dict[str, List[torch.Tensor]] = {"observations": [], "actions": [], "rewards": []}

        vae.eval()
        policy.eval()
        with torch.no_grad():
            T = features.shape[0]
            for t in range(T - 1):
                obs_t = features[t]  # [N, F]
                obs_tensor = obs_t.unsqueeze(0).to(self.device)  # [1, N, F]

                # Latent
                if len(trajectory["observations"]) == 0:
                    latent = torch.zeros(1, metadata["config"]["latent_dim"], device=self.device)
                else:
                    obs_seq = torch.stack(trajectory["observations"]).unsqueeze(0)    # [1, L, N, F]
                    act_seq = torch.stack(trajectory["actions"]).unsqueeze(0)         # [1, L, N]
                    rew_seq = torch.stack(trajectory["rewards"]).unsqueeze(0).unsqueeze(-1)  # [1, L, 1]
                    mu, logvar, _ = vae.encode(obs_seq, act_seq, rew_seq)
                    latent = vae.reparameterize(mu, logvar)

                # Policy
                weights, _ = policy.act(obs_tensor, latent, deterministic=True)
                w = weights.squeeze(0).detach().cpu().numpy()  # [N]
                w = self._normalize_weights(w)
                cash_w = max(0.0, 1.0 - float(w.sum()))

                # Returns
                p_t = prices[t].cpu().numpy()
                p_tp1 = prices[t + 1].cpu().numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    r_assets = (p_tp1 / np.where(p_t == 0, 1e-8, p_t)) - 1.0
                    r_assets = np.nan_to_num(r_assets, nan=0.0, posinf=0.0, neginf=0.0)
                port_ret = float(np.dot(w, r_assets))
                current_capital *= (1.0 + port_ret)

                results.append({
                    "date": dates[t + 1],
                    "wealth": current_capital,
                    "returns": port_ret,
                    "weights": w.tolist(),
                    "cash_weight": cash_w,
                    "model_name": metadata.get("name", "unknown"),
                    "phase": metadata.get("phase", "unknown"),
                    "trial_number": metadata.get("trial_number", "unknown"),
                    "kind": "model",
                    "strategy_type": None,
                    "num_assets": prices.shape[1],
                })

                # Update context
                reward = port_ret
                trajectory["observations"].append(obs_tensor.squeeze(0).detach())
                trajectory["actions"].append(torch.tensor(w, device=self.device))
                trajectory["rewards"].append(torch.tensor(reward, device=self.device))

                if len(trajectory["observations"]) > self.max_context:
                    for k in trajectory.keys():
                        trajectory[k] = trajectory[k][-self.max_context:]

        return pd.DataFrame(results)

    def backtest_all_models(self, model_list: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        if model_list is None:
            model_list = auto_discover_models()
        if len(model_list) == 0:
            logger.warning("No models discovered for backtesting.")
            return pd.DataFrame()

        all_df = []
        for i, m in enumerate(model_list, 1):
            logger.info(f"Backtesting model {i}/{len(model_list)}: {m.get('name')}")
            try:
                df = self.backtest_single_model(m)
                if not df.empty:
                    all_df.append(df)
                else:
                    logger.warning(f"Empty results for {m.get('name')}")
            except Exception as e:
                logger.error(f"Failed to backtest {m.get('name')}: {e}")

        return pd.concat(all_df, ignore_index=True) if all_df else pd.DataFrame()

    # -----------------------------
    # Benchmarks
    # -----------------------------
    def run_benchmark_strategy(self, strategy: BaseStrategy) -> pd.DataFrame:
        features = self.test_data["features"]
        prices = self.test_data["raw_prices"]
        dates = self.test_data["dates"]
        T, N = features.shape[0], features.shape[1]

        results: List[Dict[str, Any]] = []
        current_capital = float(self.initial_capital)
        asset_returns_history: List[np.ndarray] = []

        for t in range(T - 1):
            obs_t = features[t].cpu().numpy()  # [N, F]
            p_t = prices[t].cpu().numpy()
            p_tp1 = prices[t + 1].cpu().numpy()

            context = {
                "current_prices": p_t,
                "current_capital": current_capital,
                "trajectory_context": {"returns": asset_returns_history},
            }

            w = strategy.get_action(obs_t, context)
            if not isinstance(w, np.ndarray) or w.shape[0] != N:
                raise ValueError(f"{strategy.name}: invalid weights shape {type(w)} {getattr(w, 'shape', None)}")
            w = self._normalize_weights(w)
            cash_w = max(0.0, 1.0 - float(w.sum()))

            with np.errstate(divide="ignore", invalid="ignore"):
                r_assets = (p_tp1 / np.where(p_t == 0, 1e-8, p_t)) - 1.0
                r_assets = np.nan_to_num(r_assets, nan=0.0, posinf=0.0, neginf=0.0)

            port_ret = float(np.dot(w, r_assets))
            current_capital *= (1.0 + port_ret)

            results.append({
                "date": dates[t + 1],
                "wealth": current_capital,
                "returns": port_ret,
                "weights": w.tolist(),
                "cash_weight": cash_w,
                "model_name": f"benchmark_{strategy.name}",
                "phase": "benchmark",
                "trial_number": 0,
                "kind": "benchmark",
                "strategy_type": strategy.name,
                "num_assets": N,
            })

            asset_returns_history.append(r_assets.astype(np.float32))
            if len(asset_returns_history) > self.max_context:
                asset_returns_history = asset_returns_history[-self.max_context:]

        return pd.DataFrame(results)

    def backtest_benchmarks(self, strategy_names: Optional[List[str]] = None, exclude: Optional[List[str]] = None) -> pd.DataFrame:
        N = self.test_data["features"].shape[1]
        factory = BenchmarkFactory(num_assets=N, seed=self.seed)
        if strategy_names:
            strategies = factory.from_names(strategy_names)
        else:
            strategies = factory.create_all(exclude=exclude or [])

        all_df = []
        for name, strat in strategies.items():
            logger.info(f"Running benchmark: {name}")
            try:
                df = self.run_benchmark_strategy(strat)
                if not df.empty:
                    all_df.append(df)
            except Exception as e:
                logger.error(f"Failed benchmark {name}: {e}")
        return pd.concat(all_df, ignore_index=True) if all_df else pd.DataFrame()

    # -----------------------------
    # Summaries & Saving
    # -----------------------------
    def generate_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        def sharpe(x):
            mu = np.mean(x)
            sd = np.std(x) + 1e-8
            return mu / sd
        summary = (
            df.groupby(["model_name", "phase", "trial_number", "kind"], as_index=False)
              .agg(
                  sharpe_ratio=("returns", sharpe),
                  total_return=("wealth", lambda s: s.iloc[-1] / s.iloc[0] - 1.0 if len(s) > 1 else 0.0),
                  max_drawdown=("wealth", self._max_drawdown),
              )
              .sort_values("sharpe_ratio", ascending=False)
        )
        return summary

    def save_outputs(
        self,
        combined_df: pd.DataFrame,
        out_csv: Optional[str] = None,
        out_wide_csv: Optional[str] = None,
        also_parquet: bool = False,
    ) -> Dict[str, str]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = out_csv or (self.results_dir / f"combined_backtests_{ts}.csv").as_posix()
        out_wide_csv = out_wide_csv or (self.results_dir / f"combined_backtests_wide_{ts}.csv").as_posix()

        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(out_csv, index=False)
        logger.info(f"Long-form CSV saved to: {out_csv}")

        # Wide (wealth by date x model_name)
        wide = combined_df.pivot_table(index="date", columns="model_name", values="wealth", aggfunc="last")
        wide.to_csv(out_wide_csv)
        logger.info(f"Wide CSV saved to: {out_wide_csv}")

        if also_parquet:
            out_parq = Path(out_csv).with_suffix(".parquet").as_posix()
            out_wide_parq = Path(out_wide_csv).with_suffix(".parquet").as_posix()
            combined_df.to_parquet(out_parq, index=False)
            wide.to_parquet(out_wide_parq)
            logger.info(f"Parquet copies saved to: {out_parq}, {out_wide_parq}")

        # Save a summary CSV, too
        summary = self.generate_summary_statistics(combined_df)
        summary_csv = Path(out_csv).with_name(Path(out_csv).stem + "_summary.csv").as_posix()
        summary.to_csv(summary_csv, index=False)
        logger.info(f"Summary saved to: {summary_csv}")

        return {"long_csv": out_csv, "wide_csv": out_wide_csv, "summary_csv": summary_csv}

# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified VariBAD Model + Benchmark Backtester (day-level CSV output)")
    p.add_argument("--data-path", type=str, default="environments/data/sp500_rl_ready_cleaned.parquet")
    p.add_argument("--train-end", type=str, default="2018-12-31")
    p.add_argument("--val-end", type=str, default="2020-12-31")
    p.add_argument("--initial-capital", type=float, default=100000.0)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)

    # Models
    p.add_argument("--no-models", action="store_true", help="Skip trained-model backtests")
    p.add_argument("--search-dirs", nargs="*", default=None, help="Override model checkpoint search dirs")

    # Benchmarks
    p.add_argument("--no-benchmarks", action="store_true", help="Skip benchmark strategies")
    p.add_argument("--benchmarks", nargs="*", default=None, help="Specific benchmark names (default: all)")
    p.add_argument("--exclude-benchmarks", nargs="*", default=None, help="Benchmarks to exclude when running all")

    # Output
    p.add_argument("--output-csv", type=str, default=None, help="Long-form CSV path")
    p.add_argument("--output-wide-csv", type=str, default=None, help="Wide CSV path")
    p.add_argument("--also-parquet", action="store_true", help="Also write Parquet copies")

    return p.parse_args()

def main():
    args = parse_args()

    engine = UnifiedBacktestEngine(
        data_path=args.data_path,
        train_end=args.train_end,
        val_end=args.val_end,
        initial_capital=args.initial_capital,
        device=args.device,
        seed=args.seed,
    )
    engine.setup_test_environment()

    combined = []

    # Models
    if not args.no_models:
        model_list = auto_discover_models(args.search_dirs) if args.search_dirs is not None else auto_discover_models()
        if len(model_list) == 0:
            logger.warning("No models found; continuing without model backtests.")
        else:
            df_models = engine.backtest_all_models(model_list=model_list)
            if not df_models.empty:
                combined.append(df_models)

    # Benchmarks
    if not args.no_benchmarks:
        df_bench = engine.backtest_benchmarks(strategy_names=args.benchmarks,
                                              exclude=args.exclude_benchmarks)
        if not df_bench.empty:
            combined.append(df_bench)

    if not combined:
        logger.error("No results produced (no models/benchmarks). Nothing to save.")
        return

    combined_df = pd.concat(combined, ignore_index=True)
    paths = engine.save_outputs(
        combined_df,
        out_csv=args.output_csv,
        out_wide_csv=args.output_wide_csv,
        also_parquet=args.also_parquet,
    )

    print("\n✅ Backtesting complete.")
    print(f"   Long CSV: {paths['long_csv']}")
    print(f"   Wide  CSV: {paths['wide_csv']}")
    print(f"   Summary : {paths['summary_csv']}")

if __name__ == "__main__":
    main()
