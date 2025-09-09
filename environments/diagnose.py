import pandas as pd

def diagnose_missing(data: pd.DataFrame, freq="15T"):
    tickers = data["ticker"].unique()
    full_index = pd.date_range(start=data["date"].min(),
                               end=data["date"].max(),
                               freq=freq)

    expected = len(full_index) * len(tickers)
    actual = len(data)

    print(f"Expected rows: {expected:,}")
    print(f"Actual rows:   {actual:,}")
    print(f"Missing rows:  {expected - actual:,} "
          f"({(expected - actual)/expected:.2%} missing)")

    # Missing counts per ticker
    missing_per_ticker = {}
    for ticker in tickers:
        df_t = data[data["ticker"] == ticker]
        df_t = df_t.set_index("date").reindex(full_index)
        missing_per_ticker[ticker] = df_t["close"].isna().sum()

    missing_df = pd.DataFrame.from_dict(missing_per_ticker, orient="index", columns=["missing"])
    missing_df["missing_pct"] = missing_df["missing"] / len(full_index) * 100
    return missing_df.sort_values("missing", ascending=False)


from data_preparation import create_crypto_dataset

crypto_path = create_crypto_dataset(force_recreate=False)
df_crypto = pd.read_parquet(crypto_path)

missing_report = diagnose_missing(df_crypto, freq="15T")
print(missing_report.head(10))
