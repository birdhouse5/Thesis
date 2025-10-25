import os
import pandas as pd
import numpy as np

ROOT = r"results/final_study/final_thesis"

EXPERIMENTS = {
    "vae": {
        "path": os.path.join(ROOT, "sp500_sharpe_optimized", "vae", "sp500", "experiment_logs"),
        "file_prefix": "sp500_sharpe_optimized",
    },
    "hmm": {
        "path": os.path.join(ROOT, "sp500_hmm_sharpe_optimized", "hmm", "sp500", "experiment_logs"),
        "file_prefix": "sp500_hmm_sharpe_optimized",
    },
    "none": {
        "path": os.path.join(ROOT, "sp500_none_sharpe_optimized", "none", "sp500", "experiment_logs"),
        "file_prefix": "sp500_none_sharpe_optimized",
    },
}

INITIAL_CAPITAL = 100_000.0  # same as in MetaEnv

for exp_name, cfg in EXPERIMENTS.items():
    folder = cfg["path"]

    for file in os.listdir(folder):
        if not file.endswith("backtest.csv"):
            continue

        full_path = os.path.join(folder, file)
        print(f"Processing {full_path}")

        df = pd.read_csv(full_path)

        # 1️⃣ Add back transaction cost to get frictionless log/excess returns
        df["log_return"] = df["log_return"] + df["transaction_cost"]
        df["excess_return"] = df["excess_return"] + df["transaction_cost"]

        # 2️⃣ Recompute capital path
        df["capital"] = INITIAL_CAPITAL * np.exp(df["log_return"].cumsum())

        # 3️⃣ Save corrected (frictionless) data back to same file
        df.to_csv(full_path, index=False)

        print(f"Updated and saved {file}")

print("✅ All experiment logs converted to frictionless returns.")
