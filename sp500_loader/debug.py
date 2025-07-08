import pandas as pd
import yfinance as yf
from datetime import datetime

# Debug script to check what's wrong with the S&P 500 data

def debug_sp500_data(parquet_file='data/sp500_dataset.parquet'):
    """Debug the S&P 500 dataset to understand the price values."""
    
    print("=== DEBUGGING S&P 500 DATASET ===\n")
    
    # Load the dataset
    try:
        df = pd.read_parquet(parquet_file)
        print(f"✓ Successfully loaded dataset from {parquet_file}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print()
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # Check the index structure
    print("=== INDEX STRUCTURE ===")
    print(f"Index names: {df.index.names}")
    print(f"Number of dates: {len(df.index.get_level_values('date').unique())}")
    print(f"Number of tickers: {len(df.index.get_level_values('ticker').unique())}")
    print(f"Date range: {df.index.get_level_values('date').min()} to {df.index.get_level_values('date').max()}")
    print()
    
    # Sample some data
    print("=== SAMPLE DATA ===")
    print("First 10 rows:")
    print(df.head(10))
    print()
    
    # Check price statistics
    price_col = 'close' if 'close' in df.columns else 'adj_close'
    print(f"=== PRICE STATISTICS ({price_col}) ===")
    prices = df[price_col].dropna()
    print(f"Count: {len(prices):,}")
    print(f"Min: {prices.min():.6f}")
    print(f"Max: {prices.max():.6f}")
    print(f"Mean: {prices.mean():.6f}")
    print(f"Median: {prices.median():.6f}")
    print(f"25th percentile: {prices.quantile(0.25):.6f}")
    print(f"75th percentile: {prices.quantile(0.75):.6f}")
    print()
    
    # Sample a specific ticker
    sample_ticker = df.index.get_level_values('ticker').unique()[0]
    print(f"=== SAMPLE TICKER: {sample_ticker} ===")
    ticker_data = df.xs(sample_ticker, level='ticker').dropna(subset=[price_col])
    print(f"Number of records: {len(ticker_data)}")
    print("Last 10 records:")
    print(ticker_data.tail(10))
    print()
    
    # Compare with direct yfinance download
    print(f"=== COMPARING WITH DIRECT YFINANCE DOWNLOAD ===")
    try:
        recent_data = yf.download(sample_ticker, start='2024-01-01', end='2025-01-01', auto_adjust=False)
        print(f"Direct yfinance download for {sample_ticker}:")
        print("Columns:", recent_data.columns.tolist())
        print("Last 5 records:")
        print(recent_data[['Close', 'Adj Close']].tail())
        print()
        
        # Compare values
        if not recent_data.empty:
            latest_yf_close = recent_data['Close'].iloc[-1]
            latest_stored_close = ticker_data[price_col].iloc[-1] if not ticker_data.empty else None
            print(f"Latest yfinance Close: {latest_yf_close:.2f}")
            print(f"Latest stored {price_col}: {latest_stored_close:.6f}")
            
            if latest_stored_close and abs(latest_yf_close - latest_stored_close) > 0.01:
                print("⚠️  WARNING: Significant difference between stored and current yfinance data!")
    except Exception as e:
        print(f"Error downloading comparison data: {e}")
    
    print("\n=== DIAGNOSIS ===")
    if prices.max() < 20:
        print("🚨 ISSUE DETECTED: Maximum price is suspiciously low!")
        print("Possible causes:")
        print("1. Data was accidentally transformed (log, normalized, etc.)")
        print("2. Wrong column was stored")
        print("3. Data corruption during download/storage")
        print("4. Currency/units issue")
        
        # Check if values look like log returns
        if prices.min() > -5 and prices.max() < 5:
            print("📊 Values look like they might be log returns or normalized data")
        
        # Check if values look like split-adjusted ratios
        if prices.min() > 0 and prices.max() < 10:
            print("📊 Values might be split adjustment factors or ratios")
    else:
        print("✓ Price values look reasonable")

if __name__ == "__main__":
    debug_sp500_data()