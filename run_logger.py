# run_logger.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Iterable, Optional, Dict, Any
import csv
import json
import random

import numpy as np

# Optional: seeding for reproducibility (Python, NumPy, PyTorch if available)
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make cuDNN deterministic (slower but reproducible)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
    except Exception:
        # PyTorch not installed or not available — ignore silently
        pass


class RunLogger:
    """
    Minimal, clear experiment logger.

    Per-run files:
      - config.json      : configuration snapshot for the run
      - metrics.csv      : long-form metrics (single source of truth)
                          columns = [timestamp, split, scope, metric, value, episode, step]
      - wealth.csv       : optional time-series wealth per episode
                          columns = [timestamp, episode, t, wealth]
      - summary.csv      : one-row summary with best/final metrics
      - checkpoints/     : directory for model checkpoints (you save from main.py)

    Usage:
        run = RunLogger(run_dir, config.__dict__, name=config.exp_name)
        run.log_train_episode(ep, reward=..., sharpe=..., cum_wealth=...)
        run.log_val(ep, sharpe=..., reward=..., cum_wealth=...)
        run.log_test(sharpe=..., reward=..., cum_wealth=...)
        run.log_wealth_curve(ep, wealth_series)
        run.close()  # writes summary.csv and closes files
    """

    METRICS_HEADER = ["timestamp", "split", "scope", "metric", "value", "episode", "step"]
    WEALTH_HEADER = ["timestamp", "episode", "t", "wealth"]

    def __init__(self, run_dir: Path | str, config: Dict[str, Any], name: str = "exp") -> None:
        self.run_dir = Path(run_dir)
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        # Save config snapshot
        cfg = dict(config or {})
        cfg.setdefault("name", name)
        cfg.setdefault("seed", 42)
        cfg["run_dir"] = str(self.run_dir.resolve())
        cfg["start_time"] = datetime.utcnow().isoformat(timespec="seconds")
        (self.run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

        # Open CSVs (append-safe)
        self._metrics_f = open(self.run_dir / "metrics.csv", "a", newline="")
        self._metrics_w = csv.DictWriter(self._metrics_f, fieldnames=self.METRICS_HEADER)
        if self._metrics_f.tell() == 0:
            self._metrics_w.writeheader()

        self._wealth_f = open(self.run_dir / "wealth.csv", "a", newline="")
        self._wealth_w = csv.DictWriter(self._wealth_f, fieldnames=self.WEALTH_HEADER)
        if self._wealth_f.tell() == 0:
            self._wealth_w.writeheader()

        # Track key metrics for summary
        self.best_val_sharpe: Optional[float] = None
        self.final_test_sharpe: Optional[float] = None
        self._cfg = cfg
        self._closed = False

    # --------------- helpers ---------------

    def _now(self) -> str:
        return datetime.utcnow().isoformat(timespec="seconds")

    def _write_metric_row(
        self,
        *,
        split: str,
        scope: str,
        metric: str,
        value: float,
        episode: Optional[int] = None,
        step: Optional[int] = None,
    ) -> None:
        self._metrics_w.writerow(
            {
                "timestamp": self._now(),
                "split": split,
                "scope": scope,
                "metric": metric,
                "value": float(value),
                "episode": "" if episode is None else int(episode),
                "step": "" if step is None else int(step),
            }
        )
        self._metrics_f.flush()

    # --------------- public API ---------------

    def log_metric(
        self,
        *,
        split: str,
        scope: str,
        metric: str,
        value: float,
        episode: Optional[int] = None,
        step: Optional[int] = None,
    ) -> None:
        """Generic metric logging to metrics.csv."""
        self._write_metric_row(
            split=split, scope=scope, metric=metric, value=value, episode=episode, step=step
        )

    def log_train_episode(
        self,
        episode: int,
        *,
        reward: Optional[float] = None,
        sharpe: Optional[float] = None,
        cum_wealth: Optional[float] = None,
    ) -> None:
        """Convenience episode-level logging for the 'train' split."""
        if reward is not None:
            self._write_metric_row(split="train", scope="episode", metric="reward", value=reward, episode=episode)
        if sharpe is not None:
            self._write_metric_row(split="train", scope="episode", metric="sharpe", value=sharpe, episode=episode)
        if cum_wealth is not None:
            self._write_metric_row(
                split="train", scope="episode", metric="cum_wealth", value=cum_wealth, episode=episode
            )

    def log_val(
        self,
        episode: int,
        *,
        sharpe: Optional[float] = None,
        reward: Optional[float] = None,
        cum_wealth: Optional[float] = None,
    ) -> None:
        """Validation metrics (used for model selection)."""
        if reward is not None:
            self._write_metric_row(split="val", scope="eval", metric="reward", value=reward, episode=episode)
        if cum_wealth is not None:
            self._write_metric_row(split="val", scope="eval", metric="cum_wealth", value=cum_wealth, episode=episode)
        if sharpe is not None:
            self._write_metric_row(split="val", scope="eval", metric="sharpe", value=sharpe, episode=episode)
            if self.best_val_sharpe is None or sharpe > self.best_val_sharpe:
                self.best_val_sharpe = float(sharpe)

    def log_test(
        self,
        *,
        sharpe: Optional[float] = None,
        reward: Optional[float] = None,
        cum_wealth: Optional[float] = None,
    ) -> None:
        """Final test metrics."""
        if reward is not None:
            self._write_metric_row(split="test", scope="eval", metric="reward", value=reward)
        if cum_wealth is not None:
            self._write_metric_row(split="test", scope="eval", metric="cum_wealth", value=cum_wealth)
        if sharpe is not None:
            self._write_metric_row(split="test", scope="eval", metric="sharpe", value=sharpe)
            self.final_test_sharpe = float(sharpe)

    def log_wealth_curve(self, episode: int, wealth_series: Iterable[float]) -> None:
        """Write per-step wealth for one episode to wealth.csv."""
        for t, w in enumerate(wealth_series):
            self._wealth_w.writerow(
                {
                    "timestamp": self._now(),
                    "episode": int(episode),
                    "t": int(t),
                    "wealth": float(w),
                }
            )
        self._wealth_f.flush()

    def close(self) -> None:
        """Write summary.csv and close files."""
        if self._closed:
            return
        summary = {
            "run_dir": str(self.run_dir.resolve()),
            "name": self._cfg.get("name"),
            "seed": self._cfg.get("seed"),
            "best_val_sharpe": self.best_val_sharpe,
            "final_test_sharpe": self.final_test_sharpe,
            "end_time": datetime.utcnow().isoformat(timespec="seconds"),
        }
        with open(self.run_dir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary.keys()))
            w.writeheader()
            w.writerow(summary)

        try:
            self._metrics_f.close()
        finally:
            try:
                self._wealth_f.close()
            finally:
                self._closed = True
