"""
Crypto Coverage Discovery Scanner
Determines the maximum date range with full coverage across all tickers
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import time

BASE_URL = "https://api.binance.com/api/v3/klines"

CRYPTO_TICKERS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "NEOUSDT", "LTCUSDT",
    "QTUMUSDT", "ADAUSDT", "XRPUSDT", "IOTAUSDT", "TUSDUSDT",
    "XLMUSDT", "ONTUSDT", "TRXUSDT", "ETCUSDT", "ICXUSDT",
    "VETUSDT", "USDCUSDT", "LINKUSDT", "ONGUSDT", "HOTUSDT",
    "ZILUSDT", "FETUSDT", "ZRXUSDT", "BATUSDT", "ZECUSDT",
    "IOSTUSDT", "CELRUSDT", "DASHUSDT", "THETAUSDT", "ENJUSDT"
]


def get_ticker_date_range(symbol: str, interval: str = "15m") -> Optional[Tuple[datetime, datetime]]:
    """
    Get the earliest and latest available dates for a ticker.
    Returns (earliest_date, latest_date) or None if ticker not available.
    """
    # Get earliest data point
    try:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": 1,
            "startTime": 0  # Query from beginning of time
        }
        
        resp = requests.get(BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if not isinstance(data, list) or len(data) == 0:
            return None
        
        earliest_timestamp = data[0][0]  # Open time
        earliest_date = datetime.fromtimestamp(earliest_timestamp / 1000)
        
    except Exception as e:
        print(f"  ❌ Failed to get earliest date for {symbol}: {e}")
        return None
    
    # Small delay to respect rate limits
    time.sleep(0.1)
    
    # Get latest data point
    try:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": 1,
            "endTime": int(datetime.utcnow().timestamp() * 1000)
        }
        
        resp = requests.get(BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if not isinstance(data, list) or len(data) == 0:
            return None
        
        latest_timestamp = data[0][0]  # Open time
        latest_date = datetime.fromtimestamp(latest_timestamp / 1000)
        
    except Exception as e:
        print(f"  ❌ Failed to get latest date for {symbol}: {e}")
        return None
    
    return (earliest_date, latest_date)


def scan_coverage(tickers: list, interval: str = "15m") -> Dict:
    """
    Scan all tickers and determine the maximum coverage window.
    """
    print(f"🔍 Scanning {len(tickers)} tickers for coverage (interval: {interval})...\n")
    
    ticker_ranges = {}
    failed_tickers = []
    
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Checking {ticker}...", end=" ")
        
        date_range = get_ticker_date_range(ticker, interval)
        
        if date_range is None:
            print("❌ FAILED")
            failed_tickers.append(ticker)
        else:
            earliest, latest = date_range
            days_available = (latest - earliest).days
            ticker_ranges[ticker] = date_range
            print(f"✅ {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')} ({days_available} days)")
        
        # Rate limiting
        time.sleep(0.2)
    
    print("\n" + "="*80)
    
    if failed_tickers:
        print(f"\n⚠️  Failed tickers ({len(failed_tickers)}): {', '.join(failed_tickers)}")
        print("These tickers will be excluded from coverage calculation.\n")
    
    if not ticker_ranges:
        print("❌ No valid tickers found!")
        return {}
    
    # Calculate intersection (maximum coverage window)
    all_earliest_dates = [dates[0] for dates in ticker_ranges.values()]
    all_latest_dates = [dates[1] for dates in ticker_ranges.values()]
    
    # The intersection starts at the LATEST earliest date
    # and ends at the EARLIEST latest date
    coverage_start = max(all_earliest_dates)
    coverage_end = min(all_latest_dates)
    
    coverage_days = (coverage_end - coverage_start).days
    
    # Find which ticker is the bottleneck (has the latest start date)
    bottleneck_ticker = None
    for ticker, (earliest, latest) in ticker_ranges.items():
        if earliest == coverage_start:
            bottleneck_ticker = ticker
            break
    
    print("\n📊 COVERAGE ANALYSIS")
    print("="*80)
    print(f"✅ Valid tickers: {len(ticker_ranges)}/{len(tickers)}")
    print(f"\n📅 Maximum Coverage Window:")
    print(f"   Start: {coverage_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   End:   {coverage_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Duration: {coverage_days} days (~{coverage_days/30:.1f} months, ~{coverage_days/365:.1f} years)")
    
    if bottleneck_ticker:
        print(f"\n🔗 Bottleneck ticker: {bottleneck_ticker}")
        print(f"   (This ticker has the latest listing date)")
    
    # Calculate expected data points
    if interval == "15m":
        candles_per_day = 96  # 24 hours * 4 (15-min intervals)
    elif interval == "1h":
        candles_per_day = 24
    elif interval == "1d":
        candles_per_day = 1
    else:
        candles_per_day = None
    
    if candles_per_day:
        total_candles = coverage_days * candles_per_day
        total_rows = total_candles * len(ticker_ranges)
        print(f"\n📈 Expected Dataset Size:")
        print(f"   Candles per ticker: {total_candles:,}")
        print(f"   Total rows: {total_rows:,}")
        print(f"   Approximate API calls needed: {total_candles // 1000 * len(ticker_ranges):,}")
    
    # Show ticker availability summary
    print(f"\n📋 Ticker Availability Summary:")
    sorted_tickers = sorted(ticker_ranges.items(), key=lambda x: x[1][0])
    
    print(f"   Oldest listings:")
    for ticker, (earliest, latest) in sorted_tickers[:5]:
        print(f"      {ticker}: since {earliest.strftime('%Y-%m-%d')}")
    
    print(f"\n   Newest listings:")
    for ticker, (earliest, latest) in sorted_tickers[-5:]:
        print(f"      {ticker}: since {earliest.strftime('%Y-%m-%d')}")
    
    return {
        "valid_tickers": list(ticker_ranges.keys()),
        "failed_tickers": failed_tickers,
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "coverage_days": coverage_days,
        "bottleneck_ticker": bottleneck_ticker,
        "ticker_ranges": ticker_ranges
    }


if __name__ == "__main__":
    results = scan_coverage(CRYPTO_TICKERS, interval="15m")
    
    if results:
        print("\n" + "="*80)
        print("💡 RECOMMENDATIONS:")
        
        if results["coverage_days"] < 180:
            print("   ⚠️  Less than 6 months of coverage available.")
            print("   Consider removing newest tickers to extend the window.")
        elif results["coverage_days"] < 365:
            print("   ✅ 6-12 months available - good for initial training.")
        else:
            print("   ✅ Over 1 year available - excellent for training!")
        
        if results["failed_tickers"]:
            print(f"   ⚠️  {len(results['failed_tickers'])} tickers failed - verify these are still listed.")
        
        print("\n   Next steps:")
        print("   1. Decide if this coverage window is sufficient")
        print("   2. Consider removing bottleneck tickers for longer history")
        print("   3. Implement chunked download with this date range")