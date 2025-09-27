"""
optimize_ticker_selection.py
Find optimal subset of tickers balancing #assets vs coverage length
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# --- Candidate tickers (same as your ETF list, 30+) ---
TICKERS = [
    "SPY","QQQ","DIA","IWM","XLK","XLF","XLE","XLV","XLP","XLI",
    "XLY","XLU","XLB","XLRE","GLD","SLV","DBC","USO","UNG","TLT",
    "IEF","SHY","AGG","LQD","HYG","EFA","EEM","EWJ","EWU","EWG"
]

START_DATE = "1990-01-01"
END_DATE   = "2025-01-01"


def get_date_ranges(tickers, start_date, end_date):
    results = []
    for t in tickers:
        try:
            df = yf.download(t, start=start_date, end=end_date, progress=False)
            if df.empty:
                results.append((t, None, None))
                continue
            df = df.reset_index()
            min_date, max_date = df["Date"].min(), df["Date"].max()
            results.append((t, min_date, max_date))
        except Exception as e:
            print(f"❌ {t} failed: {e}")
            results.append((t, None, None))
    return pd.DataFrame(results, columns=["Ticker","Start","End"])


def greedy_selection(ranges, target_n):
    """Drop tickers with latest start until target_n reached."""
    selected = ranges.dropna().copy()
    while len(selected) > target_n:
        # Compute current common window
        common_start = selected["Start"].max()
        common_end   = selected["End"].min()
        # Drop ticker with latest start
        worst_idx = selected["Start"].idxmax()
        worst_ticker = selected.loc[worst_idx,"Ticker"]
        print(f"Dropping {worst_ticker} (start {selected.loc[worst_idx,'Start'].date()})")
        selected = selected.drop(worst_idx)
    # Final common window
    common_start = selected["Start"].max()
    common_end   = selected["End"].min()
    length_years = (common_end - common_start).days / 365.25
    return selected, common_start, common_end, length_years


def optimize_tradeoff(ranges, target_counts=[30,25,20,15]):
    results = []
    for n in target_counts:
        sel, start, end, years = greedy_selection(ranges, n)
        results.append({
            "num_assets": n,
            "start": start,
            "end": end,
            "coverage_years": round(years,1),
            "tickers": sel["Ticker"].tolist()
        })
    return results


def plot_tradeoff(results):
    xs = [r["num_assets"] for r in results]
    ys = [r["coverage_years"] for r in results]
    plt.figure(figsize=(7,5))
    plt.plot(xs, ys, marker="o")
    plt.gca().invert_xaxis()  # more assets = left
    for r in results:
        plt.text(r["num_assets"], r["coverage_years"],
                 f"{r['coverage_years']}y", ha="right", va="bottom")
    plt.title("Trade-off: #Assets vs Coverage Window")
    plt.xlabel("Number of assets")
    plt.ylabel("Common coverage (years)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    ranges = get_date_ranges(TICKERS, START_DATE, END_DATE)
    print("\n=== Raw Coverage ===")
    print(ranges.to_string(index=False))

    results = optimize_tradeoff(ranges, [30,25,20])
    print("\n=== Trade-off Results ===")
    for r in results:
        print(f"{r['num_assets']} assets → {r['coverage_years']} years "
              f"({r['start'].date()} → {r['end'].date()})")

    plot_tradeoff(results)
