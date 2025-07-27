#!/usr/bin/env python3
"""
Pytest Configuration and Fixtures for VariBAD Testing
Provides reusable test components and utilities
"""

import pytest
import pandas as pd
import numpy as np
import torch
import os
import sys
import shutil
import json
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the test infrastructure we created
from .test_infrastructure import (
    BaselineCapture,
    assert_dataframes_equal,
    check_dataset_integrity
)


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take several minutes)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "baseline: marks tests that establish baselines"
    )


@pytest.fixture(scope="session")
def test_config():
    """Global test configuration"""
    return {
        "random_seed": 42,
        "small_dataset_assets": 5,
        "small_dataset_days": 100,
        "medium_dataset_assets": 10, 
        "medium_dataset_days": 500,
        "tolerance": 1e-6,
        "performance_tolerance": 0.1,  # 10% performance degradation allowed
        "test_data_dir": "tests/data",
        "baseline_dir": "tests/baselines"
    }


@pytest.fixture(scope="session")
def baseline_data_path():
    """Path to the current production dataset"""
    data_path = "data/sp500_rl_ready_cleaned.parquet"
    if not os.path.exists(data_path):
        pytest.skip(f"Production dataset not found at {data_path}")
    return data_path


@pytest.fixture
def temp_data_dir():
    """Isolated temporary directory for test outputs"""
    temp_dir = tempfile.mkdtemp(prefix="varibad_test_")
    
    # Create subdirectories that mimic project structure
    (Path(temp_dir) / "data").mkdir()
    (Path(temp_dir) / "logs").mkdir() 
    (Path(temp_dir) / "checkpoints").mkdir()
    (Path(temp_dir) / "results").mkdir()
    (Path(temp_dir) / "plots").mkdir()
    (Path(temp_dir) / "config").mkdir()
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_market_data(test_config):
    """
    Deterministic sample data for consistent testing.
    Creates a small, predictable dataset with known properties.
    """
    np.random.seed(test_config["random_seed"])
    
    # Sample tickers (using realistic S&P 500 names)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ'][:test_config["small_dataset_assets"]]
    
    # Date range
    dates = pd.date_range(
        start='2020-01-01', 
        periods=test_config["small_dataset_days"], 
        freq='D'
    )
    
    data = []
    
    for ticker in tickers:
        # Generate realistic price series with known statistical properties
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, len(dates))  # 0.1% daily return, 2% volatility
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        for i, date in enumerate(dates):
            price = prices[i]
            
            # Basic OHLCV with realistic spreads
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = price * (1 + np.random.normal(0, 0.002))
            volume = np.random.randint(1000000, 10000000)
            
            data.append({
                'date': date,
                'ticker': ticker,
                'open': open_price,
                'high': max(high, price, open_price),
                'low': min(low, price, open_price),
                'close': price,
                'adj_close': price,
                'volume': volume
            })
    
    df = pd.DataFrame(data)
    
    # Add basic features that we know should exist
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    for ticker in tickers:
        ticker_mask = df['ticker'] == ticker
        ticker_data = df[ticker_mask].copy()
        
        # Returns
        ticker_data['returns'] = ticker_data['close'].pct_change()
        ticker_data['log_returns'] = np.log(ticker_data['close'] / ticker_data['close'].shift(1))
        
        # Simple technical indicators
        ticker_data['sma_5'] = ticker_data['close'].rolling(5).mean()
        ticker_data['sma_20'] = ticker_data['close'].rolling(20).mean()
        
        # Volatility
        ticker_data['volatility_5d'] = ticker_data['returns'].rolling(5).std()
        ticker_data['volatility_20d'] = ticker_data['returns'].rolling(20).std()
        
        # Update main dataframe
        df.loc[ticker_mask, ['returns', 'log_returns', 'sma_5', 'sma_20', 
                            'volatility_5d', 'volatility_20d']] = \
            ticker_data[['returns', 'log_returns', 'sma_5', 'sma_20', 
                        'volatility_5d', 'volatility_20d']].values
    
    # Add market-level features
    market_returns = df.groupby('date')['returns'].mean().reset_index()
    market_returns.columns = ['date', 'market_return']
    df = df.merge(market_returns, on='date', how='left')
    df['excess_returns'] = df['returns'] - df['market_return']
    
    # Add some normalized features for testing
    for ticker in tickers:
        ticker_mask = df['ticker'] == ticker
        ticker_data = df[ticker_mask].copy()
        
        # Normalize prices
        price_min = ticker_data['close'].min()
        price_max = ticker_data['close'].max