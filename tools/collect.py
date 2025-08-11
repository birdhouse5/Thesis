from pathlib import Path
import pandas as pd

rows = []
for run_dir in Path("runs").iterdir():
    s = run_dir / "summary.csv"
    if s.exists():
        rows.append(pd.read_csv(s))

if not rows:
    print("No summaries found in runs/.")
    raise SystemExit(0)

df = pd.concat(rows, ignore_index=True)

# Keep useful columns if present
cols = [c for c in ["run_dir","name","seed","best_val_sharpe","final_test_sharpe","end_time"] if c in df.columns]
df = df[cols]

df.sort_values("best_val_sharpe", ascending=False, inplace=True, na_position="last")
df.to_csv("experiments.csv", index=False)
print(df.head(10))
