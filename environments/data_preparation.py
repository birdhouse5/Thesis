"""
Simplified Data Preparation for VariBAD Portfolio Optimization
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from pathlib import Path
from typing import List
from datetime import datetime, timedelta
import random
import requests

logger = logging.getLogger(__name__)

# Asset lists
SP500_TICKERS = [
    'IBM', 'MSFT', 'ORCL', 'INTC', 'HPQ', 'CSCO',  # Tech
    'JPM', 'BAC', 'WFC', 'C', 'AXP',              # Financial
    'JNJ', 'PFE', 'MRK', 'ABT',                   # Healthcare
    'KO', 'PG', 'WMT', 'PEP',                     # Consumer Staples
    'XOM', 'CVX', 'COP',                          # Energy
    'GE', 'CAT', 'BA',                            # Industrials
    'HD', 'MCD',                                  # Consumer Disc
    'SO', 'D',                                    # Utilities
    'DD'                                          # Materials
]

CRYPTO_TICKERS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "NEOUSDT", "LTCUSDT",
    "QTUMUSDT", "ADAUSDT", "XRPUSDT", "IOTAUSDT", "TUSDUSDT",
    "XLMUSDT", "ONTUSDT", "TRXUSDT", "ETCUSDT", "ICXUSDT",
    "VETUSDT", "USDCUSDT", "LINKUSDT", "ONGUSDT", "HOTUSDT",
    "ZILUSDT", "ZRXUSDT", "BATUSDT", "ZECUSDT", "IOSTUSDT", 
    "CELRUSDT", "DASHUSDT", "THETAUSDT", "ENJUSDT", "FETUSDT"
]


def download_stock_data(tickers: List[str] = None, 
                       start_date: str = '1990-01-01', 
                       end_date: str = '2025-01-01') -> pd.DataFrame:
    """Download stock data using yfinance."""
    if tickers is None:
        tickers = SP500_TICKERS
    
    logger.info(f"Downloading {len(tickers)} tickers from {start_date} to {end_date}")
    
    all_data = []
    failed_tickers = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date, auto_adjust=True)
            
            if hist.empty:
                failed_tickers.append(ticker)
                continue
            
            hist = hist.reset_index()
            hist['ticker'] = ticker
            hist = hist.rename(columns={
                'Date': 'date', 'Open': 'open', 'High': 'high', 
                'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            })
            hist['adj_close'] = hist['close']
            all_data.append(hist[['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']])
            
        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")
            failed_tickers.append(ticker)
    
    if not all_data:
        raise ValueError("No data was successfully downloaded")
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    if failed_tickers:
        logger.warning(f"Failed tickers: {failed_tickers}")
    
    return combined


def download_crypto_data(tickers: List[str] = None, days: int = 92, 
                        target_rows: int = 263520) -> pd.DataFrame:
    """Download crypto data from Binance API."""
    if tickers is None:
        tickers = CRYPTO_TICKERS
    
    logger.info(f"Downloading {len(tickers)} crypto tickers, ~{days} days")
    
    # Sample random time period
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    
    all_data = []
    for ticker in tickers:
        try:
            df = _fetch_binance_klines(ticker, "15m", start, end)
            if df is None or len(df) < days * 96:  # 96 = 15min intervals per day
                logger.warning(f"Insufficient data for {ticker}")
                continue
            df['ticker'] = ticker
            all_data.append(df)
        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")
    
    if not all_data:
        raise ValueError("No crypto data downloaded successfully")
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined.iloc[:target_rows]  # Trim to target size


def _fetch_binance_klines(symbol: str, interval: str, start: datetime, end: datetime):
    """Fetch klines from Binance API."""
    url = "https://api.binance.com/api/v3/klines"
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
        if not isinstance(data, list) or len(data) == 0:
            break
        all_data.extend(data)
        if len(data) < 1000:
            break
        params["startTime"] = data[-1][6]
    
    if not all_data:
        return None
        
    df = pd.DataFrame(all_data, columns=[
        "openTime", "open", "high", "low", "close", "volume",
        "closeTime", "qav", "trades", "tbbav", "tbqav", "ignore"
    ])
    df["date"] = pd.to_datetime(df["openTime"], unit="ms")
    df = df[["date", "open", "high", "low", "close", "volume"]]
    return df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add essential technical indicators."""
    logger.info("Adding technical indicators...")
    
    results = []
    for ticker in df['ticker'].unique():
        data = df[df['ticker'] == ticker].copy().sort_values('date')
        
        # Returns
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Moving averages
        data['sma_5'] = data['close'].rolling(5).mean()
        data['sma_20'] = data['close'].rolling(20).mean()
        
        # Volatility
        data['volatility_20d'] = data['returns'].rolling(20).std()
        
        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['close'].ewm(span=12).mean()
        exp2 = data['close'].ewm(span=26).mean()
        data['macd'] = exp1 - exp2
        
        results.append(data)
    
    combined = pd.concat(results, ignore_index=True)
    
    # Market-wide features
    market_data = combined.groupby('date')['returns'].mean().reset_index()
    market_data.columns = ['date', 'market_return']
    combined = combined.merge(market_data, on='date', how='left')
    combined['excess_returns'] = combined['returns'] - combined['market_return']
    
    return combined


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize features using two simple methods."""
    logger.info("Normalizing features...")
    
    results = []
    for ticker in df['ticker'].unique():
        data = df[df['ticker'] == ticker].copy()
        
        # Method 1: MinMax for price-based features
        price_features = ['close', 'sma_5', 'sma_20']
        for feature in price_features:
            if feature in data.columns:
                min_val, max_val = data[feature].min(), data[feature].max()
                if max_val > min_val:
                    data[f'{feature}_norm'] = (data[feature] - min_val) / (max_val - min_val)
                else:
                    data[f'{feature}_norm'] = 0.5
        
        # Method 2: Z-score for other features  
        other_features = ['returns', 'log_returns', 'volatility_20d', 'macd', 'excess_returns']
        for feature in other_features:
            if feature in data.columns:
                mean_val, std_val = data[feature].mean(), data[feature].std()
                if std_val > 0:
                    data[f'{feature}_norm'] = (data[feature] - mean_val) / std_val
                else:
                    data[f'{feature}_norm'] = 0.0
        
        # RSI: scale from [0,100] to [-1,1]
        if 'rsi' in data.columns:
            data['rsi_norm'] = (data['rsi'] - 50) / 50
        
        # Volume: log transform then z-score
        if 'volume' in data.columns:
            log_vol = np.log1p(data['volume'])
            mean_val, std_val = log_vol.mean(), log_vol.std()
            if std_val > 0:
                data['volume_norm'] = (log_vol - mean_val) / std_val
            else:
                data['volume_norm'] = 0.0
        
        results.append(data)
    
    # Market feature normalization
    combined = pd.concat(results, ignore_index=True)
    if 'market_return' in combined.columns:
        mean_val, std_val = combined['market_return'].mean(), combined['market_return'].std()
        if std_val > 0:
            combined['market_return_norm'] = (combined['market_return'] - mean_val) / std_val
        else:
            combined['market_return_norm'] = 0.0
    
    return combined


def ensure_rectangular_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all date-ticker combinations exist."""
    logger.info("Ensuring rectangular structure...")
    
    all_dates = sorted(df['date'].unique())
    all_tickers = sorted(df['ticker'].unique())
    
    # Create complete index
    complete_idx = pd.MultiIndex.from_product(
        [all_dates, all_tickers], names=['date', 'ticker']
    )
    
    # Reindex and forward fill missing values
    df_rect = df.set_index(['date', 'ticker']).reindex(complete_idx).reset_index()
    
    # Forward fill within each ticker
    numeric_cols = df_rect.select_dtypes(include=[np.number]).columns
    df_rect = df_rect.sort_values(['ticker', 'date'])
    
    for ticker in all_tickers:
        mask = df_rect['ticker'] == ticker
        df_rect.loc[mask, numeric_cols] = df_rect.loc[mask, numeric_cols].fillna(method='ffill')
    
    # Fill any remaining NaNs with column medians
    df_rect[numeric_cols] = df_rect[numeric_cols].fillna(df_rect[numeric_cols].median())
    
    # Remove any rows that still have NaNs (shouldn't happen)
    df_rect = df_rect.dropna()
    
    return df_rect.sort_values(['date', 'ticker']).reset_index(drop=True)


def create_dataset(asset_class: str, output_path: str, force_recreate: bool = False) -> str:
    """Create complete RL-ready dataset."""
    output_path = Path(output_path)
    
    if output_path.exists() and not force_recreate:
        logger.info(f"Dataset exists: {output_path}")
        return str(output_path)
    
    logger.info(f"Creating {asset_class} dataset...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download data
        if asset_class == "sp500":
            raw_data = download_stock_data()
        elif asset_class == "crypto": 
            raw_data = download_crypto_data()
        else:
            raise ValueError(f"Unknown asset_class: {asset_class}")
        
        # Process pipeline
        with_indicators = add_technical_indicators(raw_data)
        normalized = normalize_features(with_indicators)
        final_data = ensure_rectangular_structure(normalized)
        
        # Validation
        expected_rows = final_data['date'].nunique() * final_data['ticker'].nunique()
        if len(final_data) != expected_rows:
            raise ValueError(f"Dataset not rectangular: {len(final_data)} != {expected_rows}")
        
        # Save
        final_data.to_parquet(output_path)
        
        logger.info(f"Dataset created: {output_path}")
        logger.info(f"Shape: {final_data.shape}")
        logger.info(f"Date range: {final_data['date'].min().date()} to {final_data['date'].max().date()}")
        logger.info(f"Tickers: {final_data['ticker'].nunique()}")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")
        raise


def load_dataset(data_path: str) -> pd.DataFrame:
    """Load dataset, creating if necessary."""
    path = Path(data_path)
    
    if not path.exists():
        # Determine asset class from path
        asset_class = "sp500" if "sp500" in str(path) else "crypto"
        create_dataset(asset_class, str(path))
    
    logger.info(f"Loading dataset: {path}")
    df = pd.read_parquet(path)
    
    # Basic validation
    if not all(col in df.columns for col in ['date', 'ticker', 'returns', 'close']):
        raise ValueError("Dataset missing required columns")
    
    normalized_features = [col for col in df.columns if col.endswith('_norm')]
    logger.info(f"Loaded dataset: {df.shape}, {len(normalized_features)} normalized features")
    
    return df