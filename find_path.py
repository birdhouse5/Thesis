import os
import pandas as pd

# Base path (relative to where the script is run)
base_path = os.path.join("results", "final_study", "final_thesis")

# Model folders to check
models = [
    "sp500_hmm_sharpe_optimized",
    "sp500_none_sharpe_optimized",
    "sp500_sharpe_optimized",
    "sp500_vae_sharpe_optimized"
]

# Loop through each model
for model in models:
    logs_path = os.path.join(base_path, model, "vae", "sp500", "experiment_logs")

    if not os.path.exists(logs_path):
        print(f"Path not found: {logs_path}")
        continue

    print(f"\n--- Headers for {model} ---")
    for file in os.listdir(logs_path):
        if file.endswith(".csv"):
            file_path = os.path.join(logs_path, file)
            try:
                df = pd.read_csv(file_path, nrows=0)  # Only read header
                print(f"{file}: {list(df.columns)}")
            except Exception as e:
                print(f"Error reading {file}: {e}")
