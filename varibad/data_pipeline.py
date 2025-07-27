#!/usr/bin/env python3
"""
Simplified data pipeline for VariBAD Portfolio Optimization
Downloads and processes S&P 500 data from scratch
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def create_sp500_constituents():
    """Create list of 30 S&P 500 companies for the study."""
    tickers = [
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
    
    constituents_df = pd.DataFrame({
        'ticker': tickers,
        'start_date': '1990-01-01',
        'end_date': '2025-01-01'
    })
    
    return constituents_df

def download_stock_data(tickers, start_date='1990-01-01', end_date='2025-01-01'):
    """Download stock data using yfinance."""
    logger.info(f"Downloading data for {len(tickers)} tickers...")
    
    all_data = []
    
    for i, ticker in enumerate(tickers):
        try:
            logger.info(f"Downloading {ticker} ({i+1}/{len(tickers)})...")
            
            # Download data
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                logger.warning(f"No data found for {ticker}")
                continue
            
            # Reset index to get date as column
            hist = hist.reset_index()
            
            # Add ticker column
            hist['ticker'] = ticker
            
            # Rename columns to match expected format
            hist = hist.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Add adj_close (same as close for simplicity)
            hist['adj_close'] = hist['close']
            
            # Select relevant columns
            hist = hist[['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]
            
            all_data.append(hist)
            
        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No data was successfully downloaded")
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Sort by date and ticker
    combined_data = combined_data.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    logger.info(f"Downloaded data: {combined_data.shape}")
    logger.info(f"Date range: {combined_data['date'].min()} to {combined_data['date'].max()}")
    logger.info(f"Tickers: {combined_data['ticker'].nunique()}")
    
    return combined_data

def add_technical_indicators(df):
    """Add basic technical indicators."""
    logger.info("Adding technical indicators...")
    
    # Sort by ticker and date
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    results = []
    
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].copy()
        
        # Basic returns
        ticker_data['returns'] = ticker_data['close'].pct_change()
        ticker_data['log_returns'] = np.log(ticker_data['close'] / ticker_data['close'].shift(1))
        
        # Simple moving averages
        ticker_data['sma_5'] = ticker_data['close'].rolling(5).mean()
        ticker_data['sma_20'] = ticker_data['close'].rolling(20).mean()
        
        # Volatility measures
        ticker_data['volatility_5d'] = ticker_data['returns'].rolling(5).std()
        ticker_data['volatility_20d'] = ticker_data['returns'].rolling(20).std()
        
        # Simple RSI approximation
        delta = ticker_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean() 
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        ticker_data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_period = 20
        ticker_data['bb_middle'] = ticker_data['close'].rolling(bb_period).mean()
        bb_std = ticker_data['close'].rolling(bb_period).std()
        ticker_data['bb_upper'] = ticker_data['bb_middle'] + (bb_std * 2)
        ticker_data['bb_lower'] = ticker_data['bb_middle'] - (bb_std * 2)
        
        # VWAP approximation
        ticker_data['vwap'] = (ticker_data['close'] * ticker_data['volume']).rolling(20).sum() / ticker_data['volume'].rolling(20).sum()
        
        results.append(ticker_data)
    
    combined = pd.concat(results, ignore_index=True)
    
    # Add market-wide features
    combined = add_market_features(combined)
    
    logger.info(f"Added technical indicators: {combined.shape}")
    return combined

def add_market_features(df):
    """Add market-wide features."""
    logger.info("Adding market features...")
    
    # Calculate market return (equal-weighted)
    market_data = df.groupby('date')['returns'].mean().reset_index()
    market_data = market_data.rename(columns={'returns': 'market_return'})
    
    # Add excess returns
    df = df.merge(market_data, on='date', how='left')
    df['excess_returns'] = df['returns'] - df['market_return']
    
    return df

def normalize_features(df):
    """Normalize features for RL training."""
    logger.info("Normalizing features...")
    
    # Features to normalize
    price_features = ['open', 'high', 'low', 'close', 'adj_close', 'sma_5', 'sma_20', 'bb_upper', 'bb_lower', 'bb_middle', 'vwap']
    bounded_features = ['rsi']  # Already bounded 0-100
    unbounded_features = ['volatility_5d', 'volatility_20d', 'volume']
    
    results = []
    
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].copy()
        
        # MinMax normalize price features
        for feature in price_features:
            if feature in ticker_data.columns:
                min_val = ticker_data[feature].min()
                max_val = ticker_data[feature].max()
                if max_val > min_val:
                    ticker_data[f'{feature}_norm'] = (ticker_data[feature] - min_val) / (max_val - min_val)
                else:
                    ticker_data[f'{feature}_norm'] = 0.0
        
        # Normalize RSI to [-1, 1]
        for feature in bounded_features:
            if feature in ticker_data.columns:
                ticker_data[f'{feature}_norm'] = (ticker_data[feature] - 50) / 50
        
        # Robust standardization for unbounded features
        for feature in unbounded_features:
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

def clean_data(df):
    """Clean the data by removing NaNs."""
    logger.info("Cleaning data...")
    
    initial_rows = len(df)
    initial_nans = df.isnull().sum().sum()
    
    logger.info(f"Initial data: {initial_rows} rows, {initial_nans} NaNs")
    
    # Remove rows where essential data is missing
    essential_columns = ['date', 'ticker', 'close', 'returns']
    df = df.dropna(subset=essential_columns)
    
    # Fill remaining NaNs with appropriate values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if col.endswith('_norm'):
            df[col] = df[col].fillna(0.0)  # Neutral value for normalized features
        elif 'rsi' in col.lower():
            df[col] = df[col].fillna(50.0)  # Neutral RSI
        elif 'volatility' in col.lower():
            df[col] = df[col].fillna(df[col].median())  # Median volatility
        else:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    final_rows = len(df)
    final_nans = df.isnull().sum().sum()
    
    logger.info(f"Cleaned data: {final_rows} rows, {final_nans} NaNs")
    logger.info(f"Removed {initial_rows - final_rows} rows")
    
    return df

def create_rl_dataset():
    """Create the complete RL-ready dataset."""
    logger.info("Creating RL-ready dataset...")
    
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    # Step 1: Get constituents
    constituents = create_sp500_constituents()
    constituents.to_csv("data/sp500_constituents.csv", index=False)
    logger.info("✅ Created constituents file")
    
    # Step 2: Download raw data
    raw_data = download_stock_data(constituents['ticker'].tolist())
    raw_data.to_parquet("data/sp500_ohlcv_dataset.parquet")
    logger.info("✅ Downloaded and saved raw OHLCV data")
    
    # Step 3: Add technical indicators
    with_indicators = add_technical_indicators(raw_data)
    with_indicators.to_parquet("data/sp500_with_indicators.parquet")
    logger.info("✅ Added technical indicators")
    
    # Step 4: Normalize features
    normalized = normalize_features(with_indicators)
    normalized.to_parquet("data/sp500_rl_ready.parquet")
    logger.info("✅ Normalized features")
    
    # Step 5: Clean data
    clean_data_final = clean_data(normalized)
    clean_data_final.to_parquet("data/sp500_rl_ready_cleaned.parquet")
    logger.info("✅ Cleaned data and saved final dataset")
    
    # Summary
    logger.info(f"Final dataset: {clean_data_final.shape}")
    logger.info(f"Date range: {clean_data_final['date'].min().date()} to {clean_data_final['date'].max().date()}")
    logger.info(f"Tickers: {clean_data_final['ticker'].nunique()}")
    logger.info(f"Features: {len(clean_data_final.columns)}")
    
    return "data/sp500_rl_ready_cleaned.parquet"

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create the dataset
    final_path = create_rl_dataset()
    print(f"Dataset created successfully: {final_path}")