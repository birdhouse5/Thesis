"""
Unified Data Module for VariBAD Portfolio Optimization
Consolidated from data.py and data_pipeline.py - single source of truth
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import random
import requests

logger = logging.getLogger(__name__)

# # S&P 500 constituents for the study (30 companies)
# SP500_TICKERS = [
#     'IBM', 'MSFT', 'ORCL', 'INTC', 'HPQ', 'CSCO',  # Tech
#     'JPM', 'BAC', 'WFC', 'C', 'AXP',              # Financial
#     'JNJ', 'PFE', 'MRK', 'ABT',                   # Healthcare
#     'KO', 'PG', 'WMT', 'PEP',                     # Consumer Staples
#     'XOM', 'CVX', 'COP',                          # Energy
#     'GE', 'CAT', 'BA',                            # Industrials
#     'HD', 'MCD',                                  # Consumer Disc
#     'SO', 'D',                                    # Utilities
#     'DD'                                          # Materials
# ]

# CRYPTO_TICKERS = [
#     "BTCUSDT", "ETHUSDT", "BNBUSDT", "NEOUSDT", "LTCUSDT",
#     "QTUMUSDT", "ADAUSDT", "XRPUSDT", "IOTAUSDT", "TUSDUSDT",
#     "XLMUSDT", "ONTUSDT", "TRXUSDT", "ETCUSDT", "ICXUSDT",
#     "VETUSDT", "USDCUSDT", "LINKUSDT", "ONGUSDT", "HOTUSDT",
#     "ZILUSDT", "FETUSDT", "ZRXUSDT", "BATUSDT", "ZECUSDT",
#     "IOSTUSDT", "CELRUSDT", "DASHUSDT", "THETAUSDT", "ENJUSDT"
# ]

# BASE_URL = "https://api.binance.com/api/v3/klines"

# def fetch_klines(symbol, interval, start, end):
#     url = BASE_URL
#     params = {
#         "symbol": symbol,
#         "interval": interval,
#         "startTime": int(start.timestamp() * 1000),
#         "endTime": int(end.timestamp() * 1000),
#         "limit": 1000
#     }
#     all_data = []
#     while True:
#         resp = requests.get(url, params=params)
#         data = resp.json()
#         if not isinstance(data, list):
#             return None
#         all_data.extend(data)
#         if len(data) < 1000:
#             break
#         params["startTime"] = data[-1][6]
#     if not all_data:
#         return None
#     df = pd.DataFrame(all_data, columns=[
#         "openTime","open","high","low","close","volume",
#         "closeTime","qav","trades","tbbav","tbqav","ignore"
#     ])
#     df["date"] = pd.to_datetime(df["openTime"], unit="ms")
#     df = df[["date","open","high","low","close","volume"]]
#     df = df.astype({"open":float,"high":float,"low":float,"close":float,"volume":float})
#     return df

# def sample_crypto(symbols, attempts=3, days=92, interval="15m", target_rows=263520):
#     """
#     Sample crypto OHLCV data for given symbols, defaults to ~92 days of 15m candles.
#     """
#     for attempt in range(attempts):
#         print(f"\n=== Sampling attempt {attempt+1} ({interval}) ===")
#         end = datetime.utcnow()
#         start_bound = datetime(2024, 4, 2)
#         max_start = end - timedelta(days=days)
#         if start_bound >= max_start:
#             start = start_bound
#         else:
#             start = start_bound + (max_start - start_bound) * random.random()
#         end = start + timedelta(days=days)

#         all_dfs, failed, illiquid = [], [], []
#         for sym in symbols:
#             df = fetch_klines(sym, interval, start, end)
#             expected_rows = days * (1440 // 15)  # 96 per day
#             if df is None or len(df) < expected_rows:
#                 print(f"  ⚠️ {sym} incomplete: {len(df) if df is not None else 'None'} rows")
#                 failed.append(sym)
#                 continue
#             if df["volume"].sum() <= 0:
#                 print(f"  ⚠️ {sym} appears illiquid (zero volume)")
#                 illiquid.append(sym)
#                 continue
#             df["ticker"] = sym
#             all_dfs.append(df)
#         if not failed and not illiquid:
#             full = pd.concat(all_dfs, ignore_index=True)
#             print(f"✅ Success: {full.shape}")
#             return full.iloc[:target_rows]
#         else:
#             print(f"Retrying due to failed/illiquid tickers: {failed+illiquid}")
#     raise RuntimeError(f"Failed after {attempts} attempts, problematic tickers: {failed+illiquid}")


# def download_stock_data(tickers: List[str] = None, 
#                        start_date: str = '1990-01-01', 
#                        end_date: str = '2025-01-01') -> pd.DataFrame:
#     """
#     Download stock data using yfinance.
    
#     Args:
#         tickers: List of ticker symbols (defaults to SP500_TICKERS)
#         start_date: Start date string
#         end_date: End date string
    
#     Returns:
#         DataFrame with OHLCV data for all tickers
#     """
#     if tickers is None:
#         tickers = SP500_TICKERS
    
#     logger.info(f" Downloading data for {len(tickers)} tickers from {start_date} to {end_date}")
    
#     all_data = []
#     failed_tickers = []
    
#     for i, ticker in enumerate(tickers):
#         try:
#             logger.info(f"  • {ticker} ({i+1}/{len(tickers)})")
            
#             stock = yf.Ticker(ticker)
#             hist = stock.history(start=start_date, end=end_date, auto_adjust=True)
            
#             if hist.empty:
#                 logger.warning(f"      No data found for {ticker}")
#                 failed_tickers.append(ticker)
#                 continue
            
#             # Reset index and add ticker
#             hist = hist.reset_index()
#             hist['ticker'] = ticker
            
#             # Standardize column names
#             hist = hist.rename(columns={
#                 'Date': 'date',
#                 'Open': 'open',
#                 'High': 'high', 
#                 'Low': 'low',
#                 'Close': 'close',
#                 'Volume': 'volume'
#             })
            
#             hist['adj_close'] = hist['close']  # Already adjusted by yfinance
            
#             # Select relevant columns
#             hist = hist[['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]
#             all_data.append(hist)
            
#         except Exception as e:
#             logger.error(f"     Failed to download {ticker}: {e}")
#             failed_tickers.append(ticker)
#             continue
    
#     if not all_data:
#         raise ValueError(" No data was successfully downloaded")
    
#     # Combine and sort
#     combined_data = pd.concat(all_data, ignore_index=True)
#     combined_data = combined_data.sort_values(['date', 'ticker']).reset_index(drop=True)
    
#     logger.info(f" Downloaded data: {combined_data.shape}")
#     logger.info(f"    Date range: {combined_data['date'].min().date()} to {combined_data['date'].max().date()}")
#     logger.info(f"    Successful tickers: {combined_data['ticker'].nunique()}")
    
#     if failed_tickers:
#         logger.warning(f"     Failed tickers: {', '.join(failed_tickers)}")
    
#     return combined_data


# def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Add comprehensive technical indicators for each ticker.
    
#     Args:
#         df: Raw OHLCV data
    
#     Returns:
#         DataFrame with technical indicators added
#     """
#     logger.info(" Adding technical indicators...")
    
#     results = []
    
#     for ticker in df['ticker'].unique():
#         ticker_data = df[df['ticker'] == ticker].copy()
#         ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
        
#         # === Basic Returns ===
#         ticker_data['returns'] = ticker_data['close'].pct_change()
#         ticker_data['log_returns'] = np.log(ticker_data['close'] / ticker_data['close'].shift(1))
        
#         # === Moving Averages ===
#         ticker_data['sma_5'] = ticker_data['close'].rolling(5).mean()
#         ticker_data['sma_20'] = ticker_data['close'].rolling(20).mean()
#         ticker_data['sma_50'] = ticker_data['close'].rolling(50).mean()
        
#         # === Volatility Measures ===
#         ticker_data['volatility_5d'] = ticker_data['returns'].rolling(5).std()
#         ticker_data['volatility_20d'] = ticker_data['returns'].rolling(20).std()
        
#         # === RSI (Relative Strength Index) ===
#         delta = ticker_data['close'].diff()
#         gain = (delta.where(delta > 0, 0)).rolling(14).mean() 
#         loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
#         rs = gain / loss
#         ticker_data['rsi'] = 100 - (100 / (1 + rs))
        
#         # === Bollinger Bands ===
#         ticker_data['bb_middle'] = ticker_data['close'].rolling(20).mean()
#         bb_std = ticker_data['close'].rolling(20).std()
#         ticker_data['bb_upper'] = ticker_data['bb_middle'] + (bb_std * 2)
#         ticker_data['bb_lower'] = ticker_data['bb_middle'] - (bb_std * 2)
#         ticker_data['bb_position'] = (ticker_data['close'] - ticker_data['bb_lower']) / (ticker_data['bb_upper'] - ticker_data['bb_lower'])
        
#         # === MACD ===
#         exp1 = ticker_data['close'].ewm(span=12).mean()
#         exp2 = ticker_data['close'].ewm(span=26).mean()
#         ticker_data['macd'] = exp1 - exp2
#         ticker_data['macd_signal'] = ticker_data['macd'].ewm(span=9).mean()
#         ticker_data['macd_histogram'] = ticker_data['macd'] - ticker_data['macd_signal']
        
#         # === Volume Weighted Average Price (VWAP) ===
#         ticker_data['vwap'] = (ticker_data['close'] * ticker_data['volume']).rolling(20).sum() / ticker_data['volume'].rolling(20).sum()
        
#         # === Price Position ===
#         ticker_data['high_low_pct'] = (ticker_data['close'] - ticker_data['low']) / (ticker_data['high'] - ticker_data['low'])
        
#         results.append(ticker_data)
    
#     combined = pd.concat(results, ignore_index=True)
    
#     # === Market-Wide Features ===
#     logger.info("🌍 Adding market-wide features...")
    
#     # Market return (equal-weighted average)
#     market_data = combined.groupby('date')['returns'].agg(['mean', 'std']).reset_index()
#     market_data = market_data.rename(columns={'mean': 'market_return', 'std': 'market_volatility'})
    
#     combined = combined.merge(market_data, on='date', how='left')
#     combined['excess_returns'] = combined['returns'] - combined['market_return']
    
#     # Market momentum (10-day rolling average of market returns)
#     market_data['market_momentum'] = market_data['market_return'].rolling(10).mean()
#     combined = combined.merge(market_data[['date', 'market_momentum']], on='date', how='left')
    
#     logger.info(f" Added technical indicators: {combined.shape}")
#     logger.info(f"    New features: {len([col for col in combined.columns if col not in ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']])}")
    
#     return combined


# def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Normalize features for RL training using appropriate methods for each feature type.
    
#     Args:
#         df: DataFrame with technical indicators
    
#     Returns:
#         DataFrame with normalized features
#     """
#     logger.info(" Normalizing features for RL training...")
    
#     results = []
    
#     # Define feature categories
#     price_features = ['open', 'high', 'low', 'close', 'adj_close', 'sma_5', 'sma_20', 'sma_50',
#                      'bb_upper', 'bb_lower', 'bb_middle', 'vwap']
    
#     bounded_features = ['rsi', 'bb_position', 'high_low_pct']  # Already bounded [0,100] or [0,1]
    
#     volatility_features = ['volatility_5d', 'volatility_20d', 'market_volatility']
    
#     volume_features = ['volume']
    
#     unbounded_features = ['macd', 'macd_signal', 'macd_histogram']
    
#     for ticker in df['ticker'].unique():
#         ticker_data = df[df['ticker'] == ticker].copy()
        
#         # === MinMax Normalization for Price Features ===
#         for feature in price_features:
#             if feature in ticker_data.columns:
#                 min_val = ticker_data[feature].min()
#                 max_val = ticker_data[feature].max()
#                 if max_val > min_val:
#                     ticker_data[f'{feature}_norm'] = (ticker_data[feature] - min_val) / (max_val - min_val)
#                 else:
#                     ticker_data[f'{feature}_norm'] = 0.5  # Neutral value if constant
        
#         # === Scale Bounded Features to [-1, 1] ===
#         for feature in bounded_features:
#             if feature in ticker_data.columns:
#                 if feature == 'rsi':
#                     # RSI: [0, 100] -> [-1, 1]
#                     ticker_data[f'{feature}_norm'] = (ticker_data[feature] - 50) / 50
#                 else:
#                     # Others: [0, 1] -> [-1, 1]
#                     ticker_data[f'{feature}_norm'] = (ticker_data[feature] - 0.5) * 2
        
#         # === Robust Standardization for Volatility ===
#         for feature in volatility_features:
#             if feature in ticker_data.columns:
#                 median_val = ticker_data[feature].median()
#                 mad = (ticker_data[feature] - median_val).abs().median()
#                 if mad > 0:
#                     ticker_data[f'{feature}_norm'] = (ticker_data[feature] - median_val) / (1.4826 * mad)
#                 else:
#                     ticker_data[f'{feature}_norm'] = 0.0
        
#         # === Log-transform + Standardization for Volume ===
#         for feature in volume_features:
#             if feature in ticker_data.columns:
#                 log_volume = np.log1p(ticker_data[feature])  # log(1 + x) to handle zeros
#                 mean_val = log_volume.mean()
#                 std_val = log_volume.std()
#                 if std_val > 0:
#                     ticker_data[f'{feature}_norm'] = (log_volume - mean_val) / std_val
#                 else:
#                     ticker_data[f'{feature}_norm'] = 0.0
        
#         # === Z-score for Unbounded Features ===
#         for feature in unbounded_features:
#             if feature in ticker_data.columns:
#                 mean_val = ticker_data[feature].mean()
#                 std_val = ticker_data[feature].std()
#                 if std_val > 0:
#                     ticker_data[f'{feature}_norm'] = (ticker_data[feature] - mean_val) / std_val
#                 else:
#                     ticker_data[f'{feature}_norm'] = 0.0
        
#         results.append(ticker_data)
    
#     combined = pd.concat(results, ignore_index=True)
    
#     # === Normalize Market-Wide Features ===
#     market_features = ['market_return', 'excess_returns', 'market_momentum']
#     for feature in market_features:
#         if feature in combined.columns:
#             mean_val = combined[feature].mean()
#             std_val = combined[feature].std()
#             if std_val > 0:
#                 combined[f'{feature}_norm'] = (combined[feature] - mean_val) / std_val
#             else:
#                 combined[f'{feature}_norm'] = 0.0
    
#     normalized_features = [col for col in combined.columns if col.endswith('_norm')]
#     logger.info(f" Normalized {len(normalized_features)} features")
    
#     return combined


# def clean_data(df: pd.DataFrame) -> pd.DataFrame:
#     """Clean data by handling missing values and ensuring rectangular structure"""
#     logger.info("🧹 Cleaning data...")
    
#     initial_rows = len(df)
#     initial_nans = df.isnull().sum().sum()
    
#     logger.info(f"    Initial: {initial_rows:,} rows, {initial_nans:,} NaN values")
    
#     # === Remove rows with missing essential data first ===
#     df = df.dropna(subset=['date', 'ticker', 'close'])
#     logger.info(f"    After removing rows with missing essentials: {len(df):,} rows")
    
#     # === Check for rectangular structure ===
#     date_counts = df.groupby('date').size()
#     expected_tickers = df['ticker'].nunique()
#     irregular_dates = date_counts[date_counts != expected_tickers]
    
#     if len(irregular_dates) > 0:
#         logger.warning(f"Found {len(irregular_dates)} dates with irregular ticker counts")
        
#         # Strategy 1: Keep only dates that have ALL tickers (most conservative)
#         valid_dates = date_counts[date_counts == expected_tickers].index
#         df_filtered = df[df['date'].isin(valid_dates)]
        
#         # Strategy 2: If we lose too much data, try filling missing tickers
#         data_loss_pct = (len(df) - len(df_filtered)) / len(df) * 100
        
#         if data_loss_pct > 20:  # If we lose more than 20% of data
#             logger.info(f"High data loss ({data_loss_pct:.1f}%), trying to fill missing tickers...")
            
#             # Create complete rectangular index
#             all_dates = sorted(df['date'].unique())
#             all_tickers = sorted(df['ticker'].unique())
#             complete_idx = pd.MultiIndex.from_product(
#                 [all_dates, all_tickers], names=['date', 'ticker']
#             )
            
#             # Reindex to ensure all date-ticker combinations exist
#             df_rect = df.set_index(['date', 'ticker']).reindex(complete_idx).reset_index()
            
#             # Forward fill missing values within each ticker (carry last observation forward)
#             numeric_cols = df_rect.select_dtypes(include=[np.number]).columns
#             df_rect = df_rect.sort_values(['ticker', 'date'])
            
#             for ticker in all_tickers:
#                 ticker_mask = df_rect['ticker'] == ticker
#                 # Forward fill within ticker, then backward fill for any remaining NaNs at start
#                 df_rect.loc[ticker_mask, numeric_cols] = (
#                     df_rect.loc[ticker_mask, numeric_cols]
#                     .fillna(method='ffill')
#                     .fillna(method='bfill')
#                 )
            
#             # If still have NaNs, use cross-sectional median for that date
#             for col in numeric_cols:
#                 if df_rect[col].isnull().any():
#                     df_rect[col] = df_rect.groupby('date')[col].transform(
#                         lambda x: x.fillna(x.median())
#                     )
            
#             # Final fallback: use overall column median
#             df_rect[numeric_cols] = df_rect[numeric_cols].fillna(df_rect[numeric_cols].median())
            
#             df = df_rect
#             logger.info(f"    After rectangular completion: {len(df):,} rows")
#         else:
#             df = df_filtered
#             logger.info(f"    After removing irregular dates: {len(df):,} rows")
    
#     # === Generate returns if missing ===
#     if 'returns' not in df.columns or df['returns'].isnull().any():
#         logger.info("    Generating returns from close prices...")
#         df = df.sort_values(['ticker', 'date'])
#         df['returns'] = df.groupby('ticker')['close'].pct_change()
    
#     # === Final cleanup ===
#     # Remove infinite values
#     df = df.replace([np.inf, -np.inf], np.nan)
    
#     # Handle remaining NaNs with appropriate defaults
#     numeric_columns = df.select_dtypes(include=[np.number]).columns
#     df = df.copy()  # Avoid SettingWithCopyWarning
    
#     for col in numeric_columns:
#         if col.endswith('_norm'):
#             # Normalized features: fill with neutral value (0)
#             df.loc[:, col] = df[col].fillna(0.0)
#         elif 'rsi' in col.lower():
#             # RSI: fill with neutral value (50)
#             df.loc[:, col] = df[col].fillna(50.0)
#         elif 'volatility' in col.lower():
#             # Volatility: fill with small positive value
#             df.loc[:, col] = df[col].fillna(0.01)
#         elif col in ['bb_position', 'high_low_pct']:
#             # Bounded ratios: fill with neutral value (0.5)
#             df.loc[:, col] = df[col].fillna(0.5)
#         elif col == 'returns':
#             # Returns: fill with 0 (no change)
#             df.loc[:, col] = df[col].fillna(0.0)
#         else:
#             # Other features: forward fill then use median
#             df.loc[:, col] = df[col].fillna(df[col].median())
    
#     # Remove any remaining rows with NaNs (should be very few)
#     df = df.dropna()
    
#     # === Remove duplicates ===
#     df = df.drop_duplicates(subset=['date', 'ticker'])
    
#     # === Ensure proper data types ===
#     df['date'] = pd.to_datetime(df['date'])
#     df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
    
#     # === Final verification ===
#     final_dates = df['date'].nunique()
#     final_tickers = df['ticker'].nunique()
#     expected_rows = final_dates * final_tickers
#     actual_rows = len(df)
    
#     final_nans = df.isnull().sum().sum()
    
#     logger.info(f"    Final: {actual_rows:,} rows, {final_nans:,} NaN values")
#     logger.info(f"    Removed: {initial_rows - actual_rows:,} rows ({100*(initial_rows - actual_rows)/initial_rows:.1f}%)")
#     logger.info(f"    Structure: {final_dates} dates × {final_tickers} tickers = {expected_rows} expected")
    
#     if actual_rows != expected_rows:
#         logger.warning(f"    ⚠️  Not perfectly rectangular: {actual_rows} vs {expected_rows} rows")
#         # Show which dates are problematic
#         remaining_date_counts = df.groupby('date').size()
#         still_irregular = remaining_date_counts[remaining_date_counts != final_tickers]
#         if len(still_irregular) > 0:
#             logger.warning(f"    Still irregular dates: {len(still_irregular)}")
#             logger.warning(f"    Sample problematic dates: {list(still_irregular.index[:5])}")
#     else:
#         logger.info(f"    ✅ Dataset is perfectly rectangular")
    
#     return df


# def create_dataset(output_path: str = "data/sp500_rl_ready_cleaned.parquet",
#                   tickers: List[str] = None,
#                   start_date: str = '1990-02-16',
#                   end_date: str = '2025-01-01',
#                   force_recreate: bool = False) -> str:
#     """
#     Create complete RL-ready dataset from scratch.
    
#     Args:
#         output_path: Where to save the final dataset
#         tickers: List of tickers (defaults to SP500_TICKERS)
#         start_date: Start date for data download
#         end_date: End date for data download
#         force_recreate: If True, recreate even if file exists
    
#     Returns:
#         Path to created dataset
#     """
#     output_path = Path(output_path)
    
#     # Check if dataset already exists
#     if output_path.exists() and not force_recreate:
#         logger.info(f" Dataset already exists: {output_path}")
#         logger.info("   Use force_recreate=True to recreate")
#         return str(output_path)
    
#     logger.info("  Creating RL-ready dataset from scratch...")
    
#     # Create data directory
#     output_path.parent.mkdir(parents=True, exist_ok=True)
    
#     try:
#         # Step 1: Download raw data
#         logger.info(" Step 1/4: Downloading stock data...")
#         raw_data = download_stock_data(tickers, start_date, end_date)
        
#         # Step 2: Add technical indicators
#         logger.info(" Step 2/4: Adding technical indicators...")
#         with_indicators = add_technical_indicators(raw_data)
        
#         # Step 3: Normalize features
#         logger.info(" Step 3/4: Normalizing features...")
#         normalized = normalize_features(with_indicators)
        
#         # Step 4: Clean data
#         logger.info(" Step 4/4: Cleaning data...")
#         cleaned = clean_data(normalized)
        
#         expected_rows = cleaned['date'].nunique() * cleaned['ticker'].nunique()
#         actual_rows = len(cleaned)
#         if actual_rows != expected_rows:
#             logger.error(f" Data not rectangular: {actual_rows} rows, expected {expected_rows}")
#             raise ValueError(f"Dataset failed rectangular validation")
#         else:
#             logger.info(f" Dataset is rectangular: {actual_rows:,} rows")

#         # Save final dataset
#         cleaned.to_parquet(output_path)
        
#         # Summary
#         file_size_mb = output_path.stat().st_size / (1024 * 1024)
#         logger.info(f"  Dataset created successfully!")
#         logger.info(f"   📁 Path: {output_path}")
#         logger.info(f"   📊 Shape: {cleaned.shape}")
#         logger.info(f"   📅 Date range: {cleaned['date'].min().date()} to {cleaned['date'].max().date()}")
#         logger.info(f"   🏢 Tickers: {cleaned['ticker'].nunique()}")
#         logger.info(f"   📈 Features: {len(cleaned.columns)}")
#         logger.info(f"   💾 Size: {file_size_mb:.1f} MB")
        
#         return str(output_path)
        
#     except Exception as e:
#         logger.error(f" Dataset creation failed: {e}")
#         raise


# def create_crypto_dataset(output_path: str = "data/crypto_rl_ready_cleaned.parquet",
#                           tickers: List[str] = None,
#                           target_rows: int = 263520,
#                           days: int = 92,
#                           interval: str = "15m",
#                           force_recreate: bool = False) -> str:
#     """
#     Create complete RL-ready crypto dataset from Binance REST API.

#     Args:
#         output_path: Path to save final dataset
#         tickers: List of tickers (default = CRYPTO_TICKERS constant)
#         target_rows: Total rows to trim to (default = stock dataset size)
#         days: Days to sample (default 92 days for 15m candles)
#         interval: Kline interval (default "15m")
#         force_recreate: If True, recreate even if file exists

#     Returns:
#         Path to created dataset
#     """
#     from pathlib import Path

#     if tickers is None:
#         tickers = CRYPTO_TICKERS   # <-- use the fixed list

#     output_path = Path(output_path)
#     if output_path.exists() and not force_recreate:
#         logger.info(f"Dataset already exists: {output_path}")
#         return str(output_path)

#     logger.info("📥 Creating RL-ready crypto dataset from Binance API...")

#     # Step 1: Sample raw crypto data
#     raw_data = sample_crypto(tickers, days=days, interval=interval, target_rows=target_rows)

#     # Step 2: Add technical indicators
#     with_indicators = add_technical_indicators(raw_data)

#     # Step 3: Normalize
#     normalized = normalize_features(with_indicators)

#     # Step 4: Clean rectangular structure
#     cleaned = clean_data(normalized)

#     # Step 5: Save
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     cleaned.to_parquet(output_path)

#     logger.info(f"✅ Crypto dataset created at {output_path}")
#     logger.info(f"   Shape: {cleaned.shape}")
#     logger.info(f"   Date range: {cleaned['date'].min().date()} → {cleaned['date'].max().date()}")
#     logger.info(f"   Tickers: {cleaned['ticker'].nunique()}")
#     logger.info(f"   Features: {len(cleaned.columns)}")

#     return str(output_path)



# def load_dataset(data_path: str = "data/sp500_rl_ready_cleaned.parquet") -> pd.DataFrame:
#     """
#     Load preprocessed dataset, creating it if it doesn't exist.
    
#     Args:
#         data_path: Path to the dataset file
    
#     Returns:
#         Preprocessed DataFrame ready for RL training
#     """
#     data_path = Path(data_path)
    
#     if not data_path.exists():
#         logger.info(f"📁 Dataset not found at {data_path}")
#         logger.info("🏗️  Creating dataset...")
#         create_dataset(str(data_path))
    
#     logger.info(f"📖 Loading dataset from {data_path}")
#     df = pd.read_parquet(data_path)
    
#     # Validate dataset
#     required_columns = ['date', 'ticker', 'returns', 'close']
#     missing_columns = [col for col in required_columns if col not in df.columns]
    
#     if missing_columns:
#         raise ValueError(f" Dataset missing required columns: {missing_columns}")
    
#     # Summary
#     normalized_features = [col for col in df.columns if col.endswith('_norm')]
    
#     logger.info(f" Dataset loaded successfully!")
#     logger.info(f"    Shape: {df.shape}")
#     logger.info(f"    Date range: {df['date'].min().date()} to {df['date'].max().date()}")
#     logger.info(f"    Tickers: {df['ticker'].nunique()}")
#     logger.info(f"    Normalized features: {len(normalized_features)}")
    
#     return df


# def get_dataset_info(data_path: str = "data/sp500_rl_ready_cleaned.parquet") -> Dict:
#     """
#     Get information about a dataset without loading it fully.
    
#     Args:
#         data_path: Path to the dataset
    
#     Returns:
#         Dictionary with dataset information
#     """
#     if not Path(data_path).exists():
#         return {"exists": False, "error": "Dataset file not found"}
    
#     try:
#         # Read just a small sample to get info
#         sample = pd.read_parquet(data_path, nrows=1000)
#         full_df = pd.read_parquet(data_path)
        
#         normalized_features = [col for col in sample.columns if col.endswith('_norm')]
#         technical_features = [col for col in sample.columns if col not in 
#                             ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]
        
#         info = {
#             "exists": True,
#             "path": data_path,
#             "file_size_mb": Path(data_path).stat().st_size / (1024 * 1024),
#             "shape": full_df.shape,
#             "date_range": {
#                 "start": str(full_df['date'].min().date()),
#                 "end": str(full_df['date'].max().date())
#             },
#             "tickers": {
#                 "count": full_df['ticker'].nunique(),
#                 "list": sorted(full_df['ticker'].unique().tolist())
#             },
#             "features": {
#                 "total": len(sample.columns),
#                 "normalized": len(normalized_features),
#                 "technical": len(technical_features)
#             },
#             "columns": sample.columns.tolist()
#         }
        
#         return info
        
#     except Exception as e:
#         return {"exists": True, "error": str(e)}
