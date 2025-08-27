#!/usr/bin/env python3
"""
plot_top_model_vs_benchmarks.py
-------------------------------------------------------------------------------
Compare ONLY the best-performing model trial against all benchmark strategies.

- Uses DAILY data (no resampling) with Month–Year x-axis ticks.
- Auto-selects the best model by:
    * final_wealth (default), or
    * sharpe (computed from daily wealth pct-change).
- Optional: restrict candidate models via --phase-filter (substring match).

Usage:
    python plot_top_model_vs_benchmarks.py \
      --input-csv results/combined_backtests_YYYYMMDD_HHMMSS.csv \
      --metric final_wealth \
      --normalize

    # Limit candidates to "phase3" models only
    python plot_top_model_vs_benchmarks.py \
      --input-csv results/combined_backtests_YYYYMMDD_HHMMSS.csv \
      --phase-filter phase3
-------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _coerce_date(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt.dt.tz_convert(None) if hasattr(dt, "dt") else dt


def _prepare_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure datetime, remove NAs, sort, and collapse duplicates (model, kind, date)."""
    df = df.copy()
    df["date"] = _coerce_date(df["date"])
    df = df.dropna(subset=["date", "wealth", "model_name", "kind"])
    df = df.sort_values(["model_name", "kind", "date"])
    df = (
        df.groupby(["model_name", "kind", "date"], as_index=False)
          .agg(wealth=("wealth", "last"))
    )
    return df


def _normalize_to_first(df: pd.DataFrame, value_col: str = "wealth") -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["model_name", "date"])
    df["wealth_index"] = (
        df.groupby("model_name", sort=False)[value_col]
          .apply(lambda s: s / (s.iloc[0] if len(s) else 1.0))
          .reset_index(level=0, drop=True)
    )
    return df


def _monthly_ticks(start: pd.Timestamp, end: pd.Timestamp):
    start_ms = start.to_period("M").to_timestamp()
    end_ms = end.to_period("M").to_timestamp()
    xticks = pd.date_range(start=start_ms, end=end_ms, freq="MS")
    labels = [d.strftime("%Y-%m") for d in xticks]
    return xticks, labels


def _choose_top_model(daily: pd.DataFrame, metric: str = "final_wealth", phase_filter: Optional[str] = None) -> str:
    """Return model_name of the top model among kind=='model'."""
    candidates = daily[daily["kind"] == "model"].copy()
    if phase_filter:
        pf = phase_filter.lower()
        candidates = candidates[candidates["model_name"].str.lower().str.contains(pf)]
        if candidates.empty:
            raise ValueError(f"No model matched phase filter '{phase_filter}'.")

    if metric == "final_wealth":
        last_vals = (
            candidates.sort_values(["model_name", "date"])
                      .groupby("model_name")["wealth"]
                      .last()
        )
        top_model = last_vals.idxmax()

    elif metric == "sharpe":
        # compute pct-change by model
        def sharpe_from_wealth(g: pd.DataFrame) -> float:
            r = g["wealth"].pct_change().dropna()
            if r.std(ddof=0) == 0 or len(r) == 0:
                return -np.inf
            return r.mean() / r.std(ddof=0)

        scores = (
            candidates.sort_values(["model_name", "date"])
                      .groupby("model_name")
                      .apply(sharpe_from_wealth)
        )
        top_model = scores.idxmax()
    else:
        raise ValueError("metric must be one of: final_wealth, sharpe")

    return str(top_model)


def _plot(df: pd.DataFrame, value_col: str, out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Full range + monthly ticks
    data_min = pd.to_datetime(df["date"].min())
    data_max = pd.to_datetime(df["date"].max())
    xticks, labels = _monthly_ticks(data_min, data_max)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=120)
    for name, g in df.sort_values(["model_name", "date"]).groupby("model_name"):
        ax.plot(g["date"], g[value_col], label=name, linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Wealth (normalized)" if value_col == "wealth_index" else "Wealth")
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot ONLY the best-performing model vs all benchmarks (daily, month ticks).")
    p.add_argument("--input-csv", type=str, required=True, help="Path to combined_backtests_*.csv (long-form)")
    p.add_argument("--metric", type=str, default="final_wealth", choices=["final_wealth", "sharpe"],
                   help="How to choose the top model")
    p.add_argument("--phase-filter", type=str, default=None, help="Substring to restrict candidate models (e.g., 'phase3')")
    p.add_argument("--normalize", action="store_true", help="Normalize series to 1.0 at first day")

    p.add_argument("--out-png", type=str, default=None, help="Output PNG path")
    p.add_argument("--out-daily-csv", type=str, default=None, help="Optional daily wide CSV export")

    return p.parse_args()


def main():
    args = parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    for col in ["date", "wealth", "model_name", "kind"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {input_csv}")

    daily = _prepare_daily(df)

    # Choose top model
    top_model = _choose_top_model(daily, metric=args.metric, phase_filter=args.phase_filter)
    print(f"Selected top model: {top_model} (metric={args.metric}, phase_filter={args.phase_filter})")

    # Keep benchmarks + the chosen model
    keep = (daily["kind"] == "benchmark") | (daily["model_name"] == top_model)
    plot_df = daily[keep].copy()

    if args.normalize:
        plot_df = _normalize_to_first(plot_df, value_col="wealth")
        value_col = "wealth_index"
    else:
        value_col = "wealth"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = Path("results") / "plots"
    out_png = Path(args.out_png) if args.out_png else plots_dir / f"top_model_vs_benchmarks_{ts}.png"
    out_daily_csv = Path(args.out_daily_csv) if args.out_daily_csv else plots_dir / f"top_model_vs_benchmarks_{ts}.csv"

    # Plot
    _plot(plot_df, value_col=value_col, out_path=out_png, title=f"Top Model vs Benchmarks (Daily) — {top_model}")

    # Optional wide CSV
    plot_df.pivot_table(index="date", columns="model_name", values=value_col, aggfunc="last").to_csv(out_daily_csv)

    print("\n✅ Plot saved.")
    print(f"   PNG : {out_png}")
    print(f"   CSV : {out_daily_csv}")


if __name__ == "__main__":
    main()
