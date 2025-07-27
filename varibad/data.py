"""
Data loading and preprocessing for VariBAD portfolio optimization
Consolidated from varibad/data_pipeline.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# S&P 500 constituents for the study (30 companies)
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


def download_stock_data(tickers: List[str] = None, 
                       start_date: str = '1990-01-01', 
                       end_date: str = '2025-01-01') -> pd.DataFrame:
    """Download stock data using yfinance."""
    
    if tickers is None:
        tickers = SP500_TICKERS
    
    logger.info(f"Downloading data for {len(tickers)} tickers...")
    
    all_data = []
    
    for i, ticker in enumerate(tickers):
        try:
            logger.info(f"Downloading {ticker} ({i+1}/{len(tickers)})...")
            
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date, auto_adjust=True)
            
            if hist.empty:
                logger.warning(f"No data found for {ticker}")
                continue
            
            # Reset index and add ticker
            hist = hist.reset_index()
            hist['ticker'] = ticker
            
            # Rename columns
            hist = hist.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            hist['adj_close'] = hist['close']  # Already adjusted
            
            # Select relevant columns
            hist = hist[['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]
            all_data.append(hist)
            
        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No data was successfully downloaded")
    
    # Combine and sort
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = combined_data.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    logger.info(f"Downloaded data: {combined_data.shape}")
    logger.info(f"Date range: {combined_data['date'].min()} to {combined_data['date'].max()}")
    
    return combined_data


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators for each ticker."""
    
    logger.info("Adding technical indicators...")
    
    results = []
    
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].copy()
        
        # Returns
        ticker_data['returns'] = ticker_data['close'].pct_change()
        ticker_data['log_returns'] = np.log(ticker_data['close'] / ticker_data['close'].shift(1))
        
        # Moving averages
        ticker_data['sma_5'] = ticker_data['close'].rolling(5).mean()
        ticker_data['sma_20'] = ticker_data['close'].rolling(20).mean()
        
        # Volatility
        ticker_data['volatility_5d'] = ticker_data['returns'].rolling(5).std()
        ticker_data['volatility_20d'] = ticker_data['returns'].rolling(20).std()
        
        # RSI
        delta = ticker_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean() 
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        ticker_data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        ticker_data['bb_middle'] = ticker_data['close'].rolling(20).mean()
        bb_std = ticker_data['close'].rolling(20).std()
        ticker_data['bb_upper'] = ticker_data['bb_middle'] + (bb_std * 2)
        ticker_data['bb_lower'] = ticker_data['bb_middle'] - (bb_std * 2)
        
        # VWAP
        ticker_data['vwap'] = (ticker_data['close'] * ticker_data['volume']).rolling(20).sum() / ticker_data['volume'].rolling(20).sum()
        
        results.append(ticker_data)
    
    combined = pd.concat(results, ignore_index=True)
    
    # Add market features
    market_data = combined.groupby('date')['returns'].mean().reset_index()
    market_data = market_data.rename(columns={'returns': 'market_return'})
    combined = combined.merge(market_data, on='date', how='left')
    combined['excess_returns'] = combined['returns'] - combined['market_return']
    
    logger.info(f"Added technical indicators: {combined.shape}")
    return combined


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize features for RL training."""
    
    logger.info("Normalizing features...")
    
    results = []
    
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].copy()
        
        # Price features (MinMax normalization)
        price_features = ['open', 'high', 'low', 'close', 'adj_close', 'sma_5', 'sma_20', 
                         'bb_upper', 'bb_lower', 'bb_middle', 'vwap']
        
        for feature in price_features:
            if feature in ticker_data.columns:
                min_val = ticker_data[feature].min()
                max_val = ticker_data[feature].max()
                if max_val > min_val:
                    ticker_data[f'{feature}_norm'] = (ticker_data[feature] - min_val) / (max_val - min_val)
                else:
                    ticker_data[f'{feature}_norm'] = 0.0
        
        # RSI normalization (to [-1, 1])
        if 'rsi' in ticker_data.columns:
            ticker_data['rsi_norm'] = (ticker_data['rsi'] - 50) / 50
        
        # Volatility features (robust standardization)
        vol_features = ['volatility_5d', 'volatility_20d', 'volume']
        for feature in vol_features:
            if feature in ticker_data.columns:
                median_val = ticker_data[feature].median()
                mad = (ticker_data[feature] - median_val).abs().median()
                if mad > 0:
                    ticker_data[f'{feature}_norm'] = (ticker_data[feature] - median_val) / (1.4826 * mad)
                else:
                    ticker_data[f'{feature}_norm'] = 0.0
        
        results.append(ticker_data)
    
    combined = pd.concat(results, ignore_index=True)
    logger.info(f"Normalized features: {combined.shape}")
    return combined


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data by handling missing values."""
    
    logger.info("Cleaning data...")
    
    initial_rows = len(df)
    initial_nans = df.isnull().sum().sum()
    
    # Remove rows with missing essential data
    essential_columns = ['date', 'ticker', 'close', 'returns']
    df = df.dropna(subset=essential_columns)
    
    # Fill remaining NaNs
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if col.endswith('_norm'):
            df[col] = df[col].fillna(0.0)
        elif 'rsi' in col.lower():
            df[col] = df[col].fillna(50.0)
        elif 'volatility' in col.lower():
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    final_rows = len(df)
    final_nans = df.isnull().sum().sum()
    
    logger.info(f"Cleaned data: {final_rows} rows, {final_nans} NaNs")
    logger.info(f"Removed {initial_rows - final_rows} rows")
    
    return df


def create_dataset(output_path: str = "data/sp500_rl_ready_cleaned.parquet",
                  force_recreate: bool = False) -> str:
    """Create complete RL-ready dataset."""
    
    output_path = Path(output_path)
    
    # Check if dataset already exists
    if output_path.exists() and not force_recreate:
        logger.info(f"Dataset already exists: {output_path}")
        return str(output_path)
    
    logger.info("Creating RL-ready dataset...")
    
    # Create data directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download data
    raw_data = download_stock_data()
    
    # Process data
    with_indicators = add_technical_indicators(raw_data)
    normalized = normalize_features(with_indicators)
    cleaned = clean_data(normalized)
    
    # Save final dataset
    cleaned.to_parquet(output_path)
    
    logger.info(f"Dataset created: {output_path}")
    logger.info(f"Final shape: {cleaned.shape}")
    logger.info(f"Date range: {cleaned['date'].min().date()} to {cleaned['date'].max().date()}")
    logger.info(f"Tickers: {cleaned['ticker'].nunique()}")
    
    return str(output_path)


def load_dataset(data_path: str) -> pd.DataFrame:
    """Load preprocessed dataset."""
    
    data_path = Path(data_path)
    
    if not data_path.exists():
        logger.info(f"Dataset not found at {data_path}, creating...")
        create_dataset(str(data_path))
    
    logger.info(f"Loading dataset from {data_path}")
    df = pd.read_parquet(data_path)
    
    logger.info(f"Loaded dataset: {df.shape}")
    logger.info(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    logger.info(f"Tickers: {df['ticker'].nunique()}")
    
    return df


if __name__ == "__main__":
    # Create dataset when run directly
    logging.basicConfig(level=logging.INFO)
    dataset_path = create_dataset()
    print(f"Dataset created: {dataset_path}")