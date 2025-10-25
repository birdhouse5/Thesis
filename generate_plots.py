import os
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # headless backend for saving only
import matplotlib.pyplot as plt

# -------------------------------
# CONFIGURATION
# -------------------------------
ROOT = r"results/final_study/final_thesis"

EXPERIMENTS = {
    "vae": {
        "path": os.path.join(ROOT, "sp500_sharpe_optimized", "vae", "sp500", "experiment_logs"),
        "file_prefix": "sp500_sharpe_optimized",
        "label": "VAE-PPO",
        "color": "#1f77b4",
    },
    "hmm": {
        "path": os.path.join(ROOT, "sp500_hmm_sharpe_optimized", "hmm", "sp500", "experiment_logs"),
        "file_prefix": "sp500_hmm_sharpe_optimized",
        "label": "HMM-PPO",
        "color": "#ff7f0e",
    },
    "none": {
        "path": os.path.join(ROOT, "sp500_none_sharpe_optimized", "none", "sp500", "experiment_logs"),
        "file_prefix": "sp500_none_sharpe_optimized",
        "label": "Vanilla PPO",
        "color": "#2ca02c",
    },
}

BENCHMARKS = {
    "buy_and_hold": {
        "path": os.path.join(ROOT, "benchmark_buy_and_hold", "benchmark", "sp500", "experiment_logs"),
        "file_prefix": "benchmark_buy_and_hold",
        "label": "Buy & Hold",
        "color": "#9467bd",
    },
    "equal_weight": {
        "path": os.path.join(ROOT, "benchmark_equal_weight", "benchmark", "sp500", "experiment_logs"),
        "file_prefix": "benchmark_equal_weight",
        "label": "Equal Weight",
        "color": "#8c564b",
    },
    "random_allocation": {
        "path": os.path.join(ROOT, "benchmark_random_allocation", "benchmark", "sp500", "experiment_logs"),
        "file_prefix": "benchmark_random_allocation",
        "label": "Random Allocation",
        "color": "#e377c2",
    },
}

OUTPUT_DIR = os.path.join(ROOT, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set(style="whitegrid", font_scale=1.2)


# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def load_csv_safe(path):
    print(f"→ Loading: {path}")
    if not os.path.exists(path):
        print(f"⚠️ Missing file: {path}")
        return None
    try:
        df = pd.read_csv(path)
        print(f"   Loaded shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return None


def downsample(df, n=5000):
    """Downsample to avoid Seaborn overload."""
    if len(df) > n:
        return df.iloc[::len(df)//n]
    return df


# -------------------------------
# PLOTS
# -------------------------------
def plot_training_curves():
    print("\n📈 Plotting training curves (diagnostic mode)...")
    try:
        import numpy as np
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        metrics = {
            "policy_loss": "Policy Loss",
            "value_loss": "Value Loss",
            "entropy": "Entropy",
            "vae_total": "VAE Total Loss",
        }

        for key, exp in EXPERIMENTS.items():
            training_path = os.path.join(exp["path"], f"{exp['file_prefix']}_training.csv")
            print(f"→ Loading: {training_path}")
            df = load_csv_safe(training_path)
            if df is None:
                continue

            if "vae_total" not in df.columns and "vae_loss" in df.columns:
                df["vae_total"] = df["vae_loss"]

            print(f"   Plotting {exp['label']}...")
            for ax, (col, title) in zip(axes, metrics.items()):
                if col in df.columns:
                    # Downsample heavily to speed up
                    step = max(1, len(df)//500)
                    ax.plot(df["cumulative_episodes"][::step],
                            df[col][::step],
                            label=exp["label"],
                            color=exp["color"],
                            linewidth=1.2)

        for ax, (_, title) in zip(axes, metrics.items()):
            ax.set_title(title)
            ax.legend()
            ax.set_xlabel("Cumulative Episodes")

        plt.tight_layout()
        outpath = os.path.join(OUTPUT_DIR, "training_curves.png")
        print("→ Saving figure...")
        plt.savefig(outpath, dpi=300)
        plt.close()
        print(f"✅ Saved: {outpath}")
    except Exception as e:
        print(f"❌ Error in training plot: {e}")



def plot_backtest_capital_curves():
    print("\n💰 Plotting backtest capital curves...")
    try:
        plt.figure(figsize=(12, 8))

        # RL models
        for key, exp in EXPERIMENTS.items():
            backtest_path = os.path.join(exp["path"], f"{exp['file_prefix']}_backtest.csv")
            df = load_csv_safe(backtest_path)
            if df is None:
                continue
            df = downsample(df)
            sns.lineplot(data=df, x="step", y="capital", label=exp["label"], color=exp["color"], linewidth=2)

        # Benchmarks
        for key, bm in BENCHMARKS.items():
            backtest_path = os.path.join(bm["path"], f"{bm['file_prefix']}_backtest.csv")
            df = load_csv_safe(backtest_path)
            if df is None:
                continue
            df = downsample(df)
            sns.lineplot(data=df, x="step", y="capital", label=bm["label"], color=bm["color"], linestyle="--")

        plt.title("Portfolio Capital Over Time (Backtest)")
        plt.xlabel("Step")
        plt.ylabel("Capital")
        plt.legend()
        plt.tight_layout()
        outpath = os.path.join(OUTPUT_DIR, "backtest_capital_curves.png")
        plt.savefig(outpath, dpi=300)
        plt.close()
        print(f"✅ Saved: {outpath}")
    except Exception as e:
        print(f"❌ Error in backtest plot: {e}")


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    print("🔍 Generating plots from experiment logs...")
    plot_training_curves()
    plot_backtest_capital_curves()
    print(f"\n🎉 All plots saved under: {OUTPUT_DIR}")
