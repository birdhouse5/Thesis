#!/usr/bin/env python3
"""
plot_backtests_daily_month_axis_v3.py
-------------------------------------------------------------------------------
Daily lines with Month–Year x-axis labels across the *full* available range.

Fixes:
- Explicit monthly ticks spanning the entire data (or a forced range)
- Robust datetime parsing (handles tz-aware -> naive)
- Collapse duplicate (model_name, kind, date) rows to avoid vertical "double" lines

Extras:
- Optional --force-range START END (e.g., 2020-01 2024-12) to lock the x-axis
- Optional faceting by kind, normalization, filtering

Usage:
    python plot_backtests_daily_month_axis_v3.py \
      --input-csv results/combined_backtests_YYYYMMDD_HHMMSS.csv \
      --normalize \
      --facet-by-kind
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
import matplotlib.dates as mdates


def _coerce_date(s: pd.Series) -> pd.Series:
    # Parse to datetime, coerce errors, keep timezone then drop to naive for matplotlib
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt.dt.tz_convert(None) if hasattr(dt, "dt") else dt


def _prepare_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure datetime, remove NAs, sort, and collapse duplicates (model, kind, date)."""
    df = df.copy()
    df["date"] = _coerce_date(df["date"])
    df = df.dropna(subset=["date", "wealth", "model_name", "kind"])

    # Sort before groupby-last to make "last" deterministic
    df = df.sort_values(["model_name", "kind", "date"])

    # Collapse duplicates at the same day per series
    df = (
        df.groupby(["model_name", "kind", "date"], as_index=False)
          .agg(wealth=("wealth", "last"))
    )
    return df


def _normalize_to_first(df: pd.DataFrame, value_col: str = "wealth") -> pd.DataFrame:
    """Add wealth_index normalized to 1.0 at first available day per series."""
    df = df.copy()
    df = df.sort_values(["model_name", "date"])
    df["wealth_index"] = (
        df.groupby("model_name", sort=False)[value_col]
          .apply(lambda s: s / (s.iloc[0] if len(s) else 1.0))
          .reset_index(level=0, drop=True)
    )
    return df


def _filter_by_kind(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    kind = kind.lower().strip()
    if kind in {"model", "models"}:
        return df[df["kind"] == "model"].copy()
    if kind in {"benchmark", "benchmarks"}:
        return df[df["kind"] == "benchmark"].copy()
    return df.copy()  # both


def _apply_date_range(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start:
        df = df[df["date"] >= pd.to_datetime(start, errors="coerce")]
    if end:
        df = df[df["date"] <= pd.to_datetime(end, errors="coerce")]
    return df


def _include_exclude(df: pd.DataFrame, include: Optional[List[str]], exclude: Optional[List[str]]) -> pd.DataFrame:
    out = df.copy()
    if include:
        inc = set([s.lower() for s in include])
        out = out[out["model_name"].str.lower().isin(inc)]
    if exclude:
        exc = set([s.lower() for s in exclude])
        out = out[~out["model_name"].str.lower().isin(exc)]
    return out


def _parse_force_range(force_range: Optional[List[str]], data_min: pd.Timestamp, data_max: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Turn --force-range START END into [start, end] Timestamps. Accept YYYY, YYYY-MM, YYYY-MM-DD."""
    if not force_range:
        return data_min, data_max

    def parse_one(s: str, default_day_end: bool = False) -> pd.Timestamp:
        # Try formats from coarse to fine
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                dt = pd.to_datetime(datetime.strptime(s, fmt))
                break
            except ValueError:
                continue
        else:
            # Fall back to pandas general parser
            dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None
        if fmt == "%Y":
            # use Jan 1 or Dec 31 depending on bound
            return dt
        if fmt == "%Y-%m":
            # use first-of-month by default; we'll floor/ceil later
            return dt
        return dt

    start_raw = parse_one(force_range[0]) if len(force_range) > 0 else None
    end_raw = parse_one(force_range[1]) if len(force_range) > 1 else None

    # Floor/ceil to day
    start = (start_raw or data_min).to_period("M").to_timestamp() if len(force_range[0]) in (4, 7) else (start_raw or data_min)
    if end_raw is not None:
        if len(force_range[1]) == 4:  # YYYY
            end = (end_raw.to_period("Y") + 1).to_timestamp() - pd.Timedelta(seconds=1)
        elif len(force_range[1]) == 7:  # YYYY-MM
            end = (end_raw.to_period("M") + 1).to_timestamp() - pd.Timedelta(seconds=1)
        else:
            end = end_raw
    else:
        end = data_max

    # Clamp to data bounds
    start = max(start, data_min)
    end = min(end, data_max)
    return start, end


def _monthly_ticks(start: pd.Timestamp, end: pd.Timestamp):
    """Generate month-start ticks and 'YYYY-MM' labels from [start, end]."""
    start_ms = start.to_period("M").to_timestamp()  # month start of start month
    end_ms = end.to_period("M").to_timestamp()
    xticks = pd.date_range(start=start_ms, end=end_ms, freq="MS")
    labels = [d.strftime("%Y-%m") for d in xticks]
    return xticks, labels


def _plot_lines(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    out_path: Path,
    legend: bool = True,
    facet_by_kind: bool = False,
    figsize: Tuple[int, int] = (12, 7),
    dpi: int = 120,
    save_svg: bool = False,
    force_range: Optional[List[str]] = None,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data_min = pd.to_datetime(df["date"].min())
    data_max = pd.to_datetime(df["date"].max())

    x_min, x_max = _parse_force_range(force_range, data_min, data_max)
    xticks, labels = _monthly_ticks(x_min, x_max)

    def _format_ax(ax):
        ax.set_title(title)
        ax.set_xlabel("Month")
        ax.set_ylabel("Wealth (normalized)" if value_col == "wealth_index" else "Wealth")
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, linestyle="--", alpha=0.4)

    if not facet_by_kind:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        for name, g in df.sort_values(["model_name", "date"]).groupby("model_name"):
            ax.plot(g["date"], g[value_col], label=name, linewidth=1.3)
        _format_ax(ax)
        if legend:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        if save_svg:
            fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)
    else:
        kinds = ["model", "benchmark"]
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(figsize[0], figsize[1] * 1.35), dpi=dpi, sharex=True)
        for ax, k in zip(axes, kinds):
            sub = df[df["kind"] == k]
            for name, g in sub.sort_values(["model_name", "date"]).groupby("model_name"):
                ax.plot(g["date"], g[value_col], label=name, linewidth=1.3)
            _format_ax(ax)
            ax.set_title(f"{title} — {k.capitalize()}s")
            if legend:
                ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        if save_svg:
            fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)


def _save_wide_daily(df: pd.DataFrame, value_col: str, out_csv: Path):
    """Daily wide CSV (date x model_name) matching the plotted values."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    wide = df.pivot_table(index="date", columns="model_name", values=value_col, aggfunc="last")
    wide.to_csv(out_csv)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot DAILY wealth with Month–Year axis for all approaches (full range).")
    p.add_argument("--input-csv", type=str, required=True, help="Path to combined_backtests_*.csv (long-form)")
    p.add_argument("--normalize", action="store_true", help="Normalize each series to 1.0 at its first day")
    p.add_argument("--kind", type=str, default="both", choices=["both", "models", "benchmarks"], help="Filter by kind")
    p.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD (optional)")
    p.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD (optional)")
    p.add_argument("--include", nargs="*", default=None, help="Only include these model_name(s)")
    p.add_argument("--exclude", nargs="*", default=None, help="Exclude these model_name(s)")
    p.add_argument("--facet-by-kind", action="store_true", help="Separate panels for models vs benchmarks")
    p.add_argument("--legend", type=str, default="true", choices=["true", "false"], help="Show legend")
    p.add_argument("--figsize", type=float, nargs=2, default=(12, 7), help="Figure size (width height)")
    p.add_argument("--dpi", type=int, default=120, help="Figure DPI")
    p.add_argument("--save-svg", action="store_true", help="Also save an SVG copy")
    p.add_argument("--force-range", nargs="+", default=None,
                   help="Force x-axis range. Examples: '2020-01 2024-12' or '2021-06-01 2023-10-31'")

    # Outputs
    p.add_argument("--out-png", type=str, default=None, help="Path for the PNG plot")
    p.add_argument("--out-daily-csv", type=str, default=None, help="Optional DAILY wide CSV export")

    return p.parse_args()


def main():
    args = parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    # Ensure required columns exist
    for col in ["date", "wealth", "model_name", "kind"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {input_csv}")

    # Prepare daily data
    daily = _prepare_daily(df)
    daily = _apply_date_range(daily, args.start_date, args.end_date)
    daily = _filter_by_kind(daily, args.kind)
    daily = _include_exclude(daily, args.include, args.exclude)

    if daily.empty:
        raise ValueError("No data to plot after filtering. Check your filters or input file.")

    if args.normalize:
        daily = _normalize_to_first(daily, value_col="wealth")
        value_col = "wealth_index"
    else:
        value_col = "wealth"

    # Defaults for outputs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = Path("results") / "plots"
    out_png = Path(args.out_png) if args.out_png else plots_dir / f"performance_daily_month_axis_{ts}.png"
    out_daily_csv = Path(args.out_daily_csv) if args.out_daily_csv else plots_dir / f"daily_wide_{ts}.csv"

    # Plot
    _plot_lines(
        df=daily,
        value_col=value_col,
        title="Performance (Daily) — Month–Year Axis",
        out_path=out_png,
        legend=(args.legend.lower() == "true"),
        facet_by_kind=args.facet_by_kind,
        figsize=(args.figsize[0], args.figsize[1]),
        dpi=args.dpi,
        save_svg=args.save_svg,
        force_range=args.force_range,
    )

    # Optional DAILY wide CSV
    _save_wide_daily(daily, value_col=value_col, out_csv=out_daily_csv)

    print("\n✅ Plot saved.")
    print(f"   PNG : {out_png}")
    if args.save_svg:
        print(f"   SVG : {out_png.with_suffix('.svg')}")
    print(f"   CSV : {out_daily_csv}")


if __name__ == "__main__":
    main()
