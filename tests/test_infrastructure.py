#!/usr/bin/env python3
"""
Test Infrastructure Setup for VariBAD Portfolio Optimization
Creates the foundation for safe refactoring with comprehensive testing
"""

import pytest
import pandas as pd
import numpy as np
import torch
import os
import shutil
import json
import hashlib
from pathlib import Path
from datetime import datetime
import tempfile
from typing import Dict, Any, Optional


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
def baseline_data_path(test_config):
    """Path to the current production dataset"""
    data_path = "data/sp500_rl_ready_cleaned.parquet"
    if not os.path.exists(data_path):
        pytest.skip(f"Production dataset not found at {data_path}")
    return data_path


@pytest.fixture
def temp_data_dir(test_config):
    """Isolated temporary directory for test outputs"""
    temp_dir = tempfile.mkdtemp(prefix="varibad_test_")
    
    # Create subdirectories
    (Path(temp_dir) / "data").mkdir()
    (Path(temp_dir) / "logs").mkdir() 
    (Path(temp_dir) / "checkpoints").mkdir()
    (Path(temp_dir) / "results").mkdir()
    
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
    
    # Sample tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'][:test_config["small_dataset_assets"]]
    
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
        ticker_data = df[df['ticker'] == ticker].copy()
        
        # Returns
        ticker_data['returns'] = ticker_data['close'].pct_change()
        ticker_data['log_returns'] = np.log(ticker_data['close'] / ticker_data['close'].shift(1))
        
        # Simple technical indicators
        ticker_data['sma_5'] = ticker_data['close'].rolling(5).mean()
        ticker_data['sma_20'] = ticker_data['close'].rolling(20).mean()
        
        # Update main dataframe
        df.loc[df['ticker'] == ticker, ['returns', 'log_returns', 'sma_5', 'sma_20']] = \
            ticker_data[['returns', 'log_returns', 'sma_5', 'sma_20']].values
    
    # Add market-level features
    market_returns = df.groupby('date')['returns'].mean().reset_index()
    market_returns.columns = ['date', 'market_return']
    df = df.merge(market_returns, on='date', how='left')
    df['excess_returns'] = df['returns'] - df['market_return']
    
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


class BaselineCapture:
    """Utility class for capturing and comparing system baselines"""
    
    def __init__(self, baseline_dir: str):
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
    def capture_dataset_baseline(self, dataset_path: str, baseline_name: str):
        """Capture dataset properties as baseline"""
        df = pd.read_parquet(dataset_path)
        
        baseline = {
            'timestamp': datetime.now().isoformat(),
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'null_counts': df.isnull().sum().to_dict(),
            'numeric_ranges': {},
            'checksum': self._calculate_dataframe_checksum(df)
        }
        
        # Capture ranges for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if not df[col].isnull().all():
                baseline['numeric_ranges'][col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std())
                }
        
        # Save baseline
        baseline_file = self.baseline_dir / f"{baseline_name}_dataset.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2)
            
        print(f"✅ Dataset baseline captured: {baseline_file}")
        return baseline
    
    def capture_training_baseline(self, log_dir: str, baseline_name: str):
        """Capture training metrics as baseline"""
        # Look for recent log files
        log_files = list(Path(log_dir).glob("varibad_pipeline_*.log"))
        if not log_files:
            print("⚠️ No training logs found for baseline capture")
            return None
            
        latest_log = max(log_files, key=os.path.getctime)
        
        # Parse basic training metrics
        metrics = self._parse_training_log(latest_log)
        
        baseline = {
            'timestamp': datetime.now().isoformat(),
            'log_file': str(latest_log),
            'metrics': metrics
        }
        
        baseline_file = self.baseline_dir / f"{baseline_name}_training.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2)
            
        print(f"✅ Training baseline captured: {baseline_file}")
        return baseline
    
    def compare_dataset_baseline(self, current_dataset_path: str, baseline_name: str, tolerance: float = 1e-6):
        """Compare current dataset against baseline"""
        baseline_file = self.baseline_dir / f"{baseline_name}_dataset.json"
        
        if not baseline_file.exists():
            pytest.fail(f"Baseline file not found: {baseline_file}")
            
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
        
        current_df = pd.read_parquet(current_dataset_path)
        current_checksum = self._calculate_dataframe_checksum(current_df)
        
        # Compare key properties
        assert current_df.shape == tuple(baseline['shape']), \
            f"Shape mismatch: {current_df.shape} vs {baseline['shape']}"
            
        assert list(current_df.columns) == baseline['columns'], \
            f"Column mismatch: {current_df.columns.tolist()} vs {baseline['columns']}"
        
        # Compare numeric ranges (with tolerance)
        for col, expected_range in baseline['numeric_ranges'].items():
            if col in current_df.columns:
                actual_mean = float(current_df[col].mean())
                expected_mean = expected_range['mean']
                
                if abs(actual_mean - expected_mean) > tolerance:
                    pytest.fail(f"Mean mismatch for {col}: {actual_mean} vs {expected_mean}")
        
        print(f"✅ Dataset comparison passed for {baseline_name}")
        return True
    
    def _calculate_dataframe_checksum(self, df: pd.DataFrame) -> str:
        """Calculate deterministic checksum for dataframe"""
        # Sort to ensure consistent ordering
        df_sorted = df.sort_values(list(df.columns)).reset_index(drop=True)
        
        # Convert to string representation and hash
        content = df_sorted.to_csv(index=False)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _parse_training_log(self, log_path: Path) -> Dict[str, Any]:
        """Extract basic metrics from training log"""
        metrics = {
            'episodes_found': 0,
            'iterations_found': 0,
            'final_reward': None,
            'errors': 0,
            'warnings': 0
        }
        
        try:
            with open(log_path, 'r') as f:
                for line in f:
                    if "Episode reward:" in line:
                        metrics['episodes_found'] += 1
                        # Try to extract the reward value
                        try:
                            reward = float(line.split("Episode reward:")[1].split()[0])
                            metrics['final_reward'] = reward
                        except:
                            pass
                    
                    if "Iteration" in line and ":" in line:
                        metrics['iterations_found'] += 1
                    
                    if "ERROR" in line or "Error" in line:
                        metrics['errors'] += 1
                    
                    if "WARNING" in line or "Warning" in line:
                        metrics['warnings'] += 1
        
        except Exception as e:
            print(f"⚠️ Error parsing log file: {e}")
        
        return metrics


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