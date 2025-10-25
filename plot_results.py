import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.style.use("seaborn-v0_8-paper")

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = "results/final_study/final_thesis"
EXPERIMENTS = {
    "vae": {
        "path": os.path.join(ROOT, "sp500_sharpe_optimized", "vae", "sp500", "experiment_logs"),
        "label": "VAE-PPO",
        "color": "#1f77b4",
    },
    "hmm": {
        "path": os.path.join(ROOT, "sp500_hmm_sharpe_optimized", "hmm", "sp500", "experiment_logs"),
        "label": "HMM-PPO",
        "color": "#ff7f0e",
    },
    "none": {
        "path": os.path.join(ROOT, "sp500_none_sharpe_optimized", "none", "sp500", "experiment_logs"),
        "label": "Vanilla PPO",
        "color": "#2ca02c",
    },
}
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def find_file(base_path, keyword):
    for f in os.listdir(base_path):
        if keyword in f and f.endswith(".csv"):
            return os.path.join(base_path, f)
    raise FileNotFoundError(f"❌ No CSV file containing '{keyword}' found in {base_path}")

def load_csv(base_path, keyword):
    path = find_file(base_path, keyword)
    print(f"✅ Found file: {os.path.basename(path)}")
    return pd.read_csv(path)

def rolling_sharpe(df, window=60):
    if "excess_return" not in df.columns:
        raise ValueError("excess_return column missing from dataframe")
    ret = df["excess_return"]
    roll_mean = ret.rolling(window).mean()
    roll_std = ret.rolling(window).std()
    return roll_mean / roll_std

def compute_drawdown(capital):
    peak = capital.cummax()
    dd = (capital - peak) / peak
    return dd.min()

def annualize_return(ret_series, periods_per_year=252):
    mean_daily = ret_series.mean()
    return (1 + mean_daily) ** periods_per_year - 1

def annualize_volatility(ret_series, periods_per_year=252):
    return ret_series.std() * np.sqrt(periods_per_year)

def sortino_ratio(ret_series, rf=0.0, periods_per_year=252):
    downside = ret_series[ret_series < rf]
    downside_std = downside.std() * np.sqrt(periods_per_year)
    if downside_std == 0:
        return np.nan
    return (annualize_return(ret_series, periods_per_year) - rf) / downside_std

def calmar_ratio(ret_series, capital):
    dd = compute_drawdown(capital)
    if dd == 0:
        return np.nan
    return annualize_return(ret_series) / abs(dd)

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
data = {}
for key, meta in EXPERIMENTS.items():
    print(f"\nLoading {key.upper()} data from {meta['path']}")
    backtest = load_csv(meta["path"], "backtest")
    validation = load_csv(meta["path"], "validation")
    data[key] = {"backtest": backtest, "validation": validation}

# -----------------------------------------------------------------------------
# Compute metrics summary
# -----------------------------------------------------------------------------
summary = []
for key, meta in EXPERIMENTS.items():
    df = data[key]["backtest"].copy()
    ret = df["excess_return"].dropna()
    capital = df["capital"].dropna()

    ann_ret = annualize_return(ret)
    ann_vol = annualize_volatility(ret)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    sortino = sortino_ratio(ret)
    calmar = calmar_ratio(ret, capital)
    max_dd = compute_drawdown(capital)
    turnover = df["turnover"].mean() if "turnover" in df.columns else np.nan
    tx_cost = df["transaction_cost"].mean() if "transaction_cost" in df.columns else np.nan

    summary.append({
        "Model": meta["label"],
        "Annual Return": ann_ret,
        "Volatility": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max Drawdown": max_dd,
        "Turnover": turnover,
        "Transaction Cost": tx_cost,
    })

summary_df = pd.DataFrame(summary)
summary_df.set_index("Model", inplace=True)
summary_df.to_csv(os.path.join(OUTDIR, "metrics_summary.csv"))

# -----------------------------------------------------------------------------
# Export LaTeX table
# -----------------------------------------------------------------------------
latex_table = summary_df.to_latex(
    float_format="%.3f",
    caption="Backtest performance comparison across models (S\\&P 500, test set).",
    label="tab:results-summary",
    column_format="lcccccccc",
    escape=False
)
with open(os.path.join(OUTDIR, "metrics_summary.tex"), "w") as f:
    f.write(latex_table)

# -----------------------------------------------------------------------------
# (Keep previous plots)
# -----------------------------------------------------------------------------
plt.figure(figsize=(7, 4))
for key, meta in EXPERIMENTS.items():
    df = data[key]["backtest"]
    grouped = df.groupby("step").agg({"capital": "mean"})
    plt.plot(grouped.index, grouped["capital"], label=meta["label"], lw=1.8, color=meta["color"])
plt.xlabel("Step")
plt.ylabel("Portfolio Value")
plt.title("Cumulative Capital Growth (Test Set)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "capital_growth.pdf"))
plt.close()

plt.figure(figsize=(7, 4))
for key, meta in EXPERIMENTS.items():
    df = data[key]["backtest"]
    rs = rolling_sharpe(df)
    plt.plot(df["step"], rs, label=meta["label"], lw=1.5, color=meta["color"])
plt.xlabel("Step")
plt.ylabel("Rolling Sharpe (60d)")
plt.title("Rolling 60-Day Sharpe Ratios")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rolling_sharpe.pdf"))
plt.close()

vae_backtest = data["vae"]["backtest"]
latent_cols = [c for c in vae_backtest.columns if c.startswith("latent_")]
if latent_cols:
    X = vae_backtest[latent_cols].dropna().values
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    vol = vae_backtest["excess_return"].rolling(20).std().iloc[:len(Z)]

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=vol, cmap="viridis", s=10, alpha=0.8)
    plt.colorbar(sc, label="Rolling Volatility (20d)")
    plt.title("VAE Latent Regime Embeddings (PCA Projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "latent_pca_regimes.pdf"))
    plt.close()

print("\n✅ All figures and tables saved to:", os.path.abspath(OUTDIR))


from scipy import stats

# -----------------------------------------------------------------------------
# Significance testing (Sharpe & Return differences)
# -----------------------------------------------------------------------------
def welch_ttest(x, y):
    """Welch's t-test for unequal variances."""
    t_stat, p_val = stats.ttest_ind(x, y, equal_var=False, nan_policy='omit')
    return t_stat, p_val

def bootstrap_diff(metric_a, metric_b, n_boot=10000):
    """Nonparametric bootstrap CI for mean difference."""
    diffs = []
    n = min(len(metric_a), len(metric_b))
    for _ in range(n_boot):
        sa = np.random.choice(metric_a, n, replace=True)
        sb = np.random.choice(metric_b, n, replace=True)
        diffs.append(np.mean(sa) - np.mean(sb))
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return np.mean(diffs), (ci_low, ci_high)

print("\nPerforming pairwise statistical significance tests...")

pairs = [("vae", "hmm"), ("vae", "none"), ("hmm", "none")]
results = []
for a, b in pairs:
    df_a, df_b = data[a]["backtest"], data[b]["backtest"]

    # Use daily excess returns
    ret_a, ret_b = df_a["excess_return"].dropna(), df_b["excess_return"].dropna()

    # Welch t-test on raw excess returns
    t_stat, p_val = welch_ttest(ret_a, ret_b)

    # Bootstrap mean difference
    mean_diff, (ci_lo, ci_hi) = bootstrap_diff(ret_a.values, ret_b.values)

    # Compute Sharpe differences
    sharpe_a = annualize_return(ret_a) / annualize_volatility(ret_a)
    sharpe_b = annualize_return(ret_b) / annualize_volatility(ret_b)
    sharpe_diff = sharpe_a - sharpe_b

    results.append({
        "Comparison": f"{EXPERIMENTS[a]['label']} – {EXPERIMENTS[b]['label']}",
        "Mean Diff": mean_diff,
        "95% CI Low": ci_lo,
        "95% CI High": ci_hi,
        "t-stat": t_stat,
        "p-value": p_val,
        "Sharpe Δ": sharpe_diff,
    })

# Convert to DataFrame and save
sig_df = pd.DataFrame(results)
sig_df.to_csv(os.path.join(OUTDIR, "significance_tests.csv"), index=False)

# Export LaTeX
latex_sig = sig_df.to_latex(
    float_format="%.4f",
    caption="Pairwise significance tests for mean return and Sharpe ratio differences.",
    label="tab:significance-tests",
    index=False,
    escape=False
)
with open(os.path.join(OUTDIR, "significance_tests.tex"), "w", encoding="utf-8") as f:
    f.write(latex_sig)


print("✅ Significance testing complete. Results saved to figures/")