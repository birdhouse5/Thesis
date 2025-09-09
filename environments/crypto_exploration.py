import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

BASE_URL = "https://api.binance.com/api/v3/klines"

def fetch_klines(symbol, interval, start, end):
    url = BASE_URL
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(start.timestamp() * 1000),
        "endTime": int(end.timestamp() * 1000),
        "limit": 1000
    }
    all_data = []
    while True:
        resp = requests.get(url, params=params)
        data = resp.json()
        if not isinstance(data, list):
            return None  # API error
        all_data.extend(data)
        if len(data) < 1000:
            break
        params["startTime"] = data[-1][6]  # last closeTime as next start
    if not all_data:
        return None
    df = pd.DataFrame(all_data, columns=[
        "openTime","open","high","low","close","volume",
        "closeTime","qav","trades","tbbav","tbqav","ignore"
    ])
    df["date"] = pd.to_datetime(df["openTime"], unit="ms")
    df = df[["date","open","high","low","close","volume"]]
    df = df.astype({"open":float,"high":float,"low":float,"close":float,"volume":float})
    return df

def sample_crypto(symbols, attempts=3, days=92, interval="15m", target_rows=263520):
    """
    Sample crypto OHLCV data for given symbols.

    - interval: now "15m" instead of "1m"
    - days: defaults to ~92 to hit 263,520 rows with 30 tickers
    """
    for attempt in range(attempts):
        print(f"\n=== Sampling attempt {attempt+1} ({interval}) ===")
        end = datetime.utcnow()
        # pick random start in overlap [2024-04-02, today - days]
        start_bound = datetime(2024, 4, 2)
        max_start = end - timedelta(days=days)
        if start_bound >= max_start:
            start = start_bound
        else:
            start = start_bound + (max_start - start_bound) * random.random()
        end = start + timedelta(days=days)

        all_dfs, failed, illiquid = [], [], []
        for sym in symbols:
            df = fetch_klines(sym, interval, start, end)
            expected_rows = days * (1440 // 15)  # 96 candles/day
            if df is None or len(df) < expected_rows:
                print(f"  ⚠️ {sym} incomplete: {len(df) if df is not None else 'None'} rows")
                failed.append(sym)
                continue
            if df["volume"].sum() <= 0:
                print(f"  ⚠️ {sym} appears illiquid (zero volume)")
                illiquid.append(sym)
                continue
            df["ticker"] = sym
            all_dfs.append(df)
        if not failed and not illiquid:
            full = pd.concat(all_dfs, ignore_index=True)
            print(f"✅ Success: {full.shape}")
            return full.iloc[:target_rows]
        else:
            print(f"Retrying due to failed/illiquid tickers: {failed+illiquid}")
    raise RuntimeError(f"Failed after {attempts} attempts, problematic tickers: {failed+illiquid}")

# Example tickers
symbols = [
    "USDCUSDT","ETHUSDT","SOLUSDT","BTCUSDT","WLDUSDT","DOGEUSDT","XRPUSDT",
    "FDUSDUSDT","ENAUSDT","BNBUSDT","TRXUSDT","ADAUSDT","SUIUSDT","PEPEUSDT",
    "LINKUSDT","BONKUSDT","AVAXUSDT","BCHUSDT","LTCUSDT","SEIUSDT","ARKMUSDT",
    "ARBUSDT","XLMUSDT","DOTUSDT","UNIUSDT","AAVEUSDT","OGUSDT","HBARUSDT",
    "PAXGUSDT","NMRUSDT"
]

# Usage:
df_sample = sample_crypto(symbols, interval="15m")
print(df_sample.head())
