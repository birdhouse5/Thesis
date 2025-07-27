#!/usr/bin/env python3
"""
Updated pytest configuration with correct data paths
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
from tests.test_infrastructure import (
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
    """Path to the current production dataset - FIXED PATH"""
    # Try multiple possible locations
    possible_paths = [
        "data/sp500_rl_ready_cleaned.parquet",  # Main location
        "varibad/data/sp500_rl_ready_cleaned.parquet",  # Alternative location
        Path("data") / "sp500_rl_ready_cleaned.parquet"  # Pathlib version
    ]
    
    for data_path in possible_paths:
        if Path(data_path).exists():
            print(f"✓ Found production dataset at: {data_path}")
            return str(data_path)
    
    # If no data found, try to create it
    print("⚠️ Production dataset not found, attempting to create...")
    try:
        # Try to create data using the pipeline
        import subprocess
        import sys
        
        result = subprocess.run([
            sys.executable, 'varibad/main.py', '--mode', 'data_only'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # Check again for the created data
            for data_path in possible_paths:
                if Path(data_path).exists():
                    print(f"✅ Created and found dataset at: {data_path}")
                    return str(data_path)
        
        print(f"❌ Data creation failed: {result.stderr}")
        
    except Exception as e:
        print(f"❌ Error creating data: {e}")
    
    pytest.skip(f"Production dataset not found at any expected location: {possible_paths}")


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
        price_max = ticker_data['close'].max()
        if price_max > price_min:
            ticker_data['close_norm'] = (ticker_data['close'] - price_min) / (price_max - price_min)
        else:
            ticker_data['close_norm'] = 0.0
            
        df.loc[ticker_mask, 'close_norm'] = ticker_data['close_norm'].values
    
    return df


@pytest.fixture(scope="session") 
def reference_technical_indicators():
    """
    Known-good values for technical indicator validation.
    Hand-calculated or from trusted external libraries.
    """
    return {
        'rsi_sample': {
            'prices': [44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 47.25, 47.32, 47.20, 47.56, 47.90, 48.10, 48.28],
            'expected_rsi_14': 70.46  # Known RSI value for this sequence
        },
        'sma_sample': {
            'prices': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'expected_sma_5': [np.nan, np.nan, np.nan, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        },
        'macd_sample': {
            'prices': list(range(1, 50)),  # Simple increasing sequence
            'expected_properties': {
                'macd_positive': True,  # Should be positive for increasing prices
                'signal_follows_macd': True  # Signal should follow MACD direction
            }
        }
    }


@pytest.fixture(scope="session")
def baseline_capture(test_config):
    """Baseline capture utility"""
    return BaselineCapture(test_config["baseline_dir"])


# Utility functions for tests
def assert_dataframes_equal(df1: pd.DataFrame, df2: pd.DataFrame, tolerance: float = 1e-6, ignore_index: bool = True):
    """Compare two dataframes with tolerance for floating point differences"""
    if ignore_index:
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
    
    # Shape check
    assert df1.shape == df2.shape, f"Shape mismatch: {df1.shape} vs {df2.shape}"
    
    # Column check
    assert list(df1.columns) == list(df2.columns), f"Column mismatch: {df1.columns.tolist()} vs {df2.columns.tolist()}"
    
    # Content check with tolerance
    for col in df1.columns:
        if df1[col].dtype in ['float64', 'float32']:
            # Numeric comparison with tolerance
            mask = ~(df1[col].isna() & df2[col].isna())  # Ignore positions where both are NaN
            if mask.any():
                diff = abs(df1.loc[mask, col] - df2.loc[mask, col])
                max_diff = diff.max()
                assert max_diff <= tolerance, f"Column {col} differs by {max_diff} > {tolerance}"
        else:
            # Exact comparison for non-numeric
            pd.testing.assert_series_equal(df1[col], df2[col], check_names=False)


def check_dataset_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    """Check basic dataset integrity and return diagnostics"""
    diagnostics = {
        'shape': df.shape,
        'null_counts': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'date_range': None,
        'tickers': None,
        'issues': []
    }
    
    # Date range check
    if 'date' in df.columns:
        diagnostics['date_range'] = {
            'start': str(df['date'].min()),
            'end': str(df['date'].max()),
            'unique_dates': df['date'].nunique()
        }
    
    # Ticker check
    if 'ticker' in df.columns:
        diagnostics['tickers'] = {
            'unique_tickers': df['ticker'].nunique(),
            'ticker_list': sorted(df['ticker'].unique().tolist())
        }
    
    # Common issues
    if diagnostics['duplicate_rows'] > 0:
        diagnostics['issues'].append(f"Found {diagnostics['duplicate_rows']} duplicate rows")
    
    if any(count > df.shape[0] * 0.5 for count in diagnostics['null_counts'].values()):
        high_null_cols = [col for col, count in diagnostics['null_counts'].items() 
                         if count > df.shape[0] * 0.5]
        diagnostics['issues'].append(f"High null counts in columns: {high_null_cols}")
    
    return diagnostics