#!/usr/bin/env python3
"""
Phase 1 Tests: Data Pipeline Consolidation
Tests for safely consolidating and simplifying data processing
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import time
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import test infrastructure
from tests.conftest import (
    assert_dataframes_equal, 
    check_dataset_integrity,
    BaselineCapture
)


class TestDataPipelineBaseline:
    """Test current data pipeline to establish baseline"""
    
    def test_production_dataset_exists(self, baseline_data_path):
        """Verify production dataset exists and is valid"""
        assert os.path.exists(baseline_data_path), f"Production dataset not found: {baseline_data_path}"
        
        # Basic file checks
        assert os.path.getsize(baseline_data_path) > 1000000, "Dataset file too small"
        
        # Can be loaded
        df = pd.read_parquet(baseline_data_path)
        assert len(df) > 10000, "Dataset has too few rows"
        assert len(df.columns) > 10, "Dataset has too few columns"
        
        print(f"✅ Production dataset valid: {df.shape}")
    
    def test_dataset_integrity(self, baseline_data_path):
        """Test dataset integrity and capture baseline properties"""
        df = pd.read_parquet(baseline_data_path)
        
        # Run integrity checks
        diagnostics = check_dataset_integrity(df)
        
        # Basic requirements
        assert 'date' in df.columns, "Dataset missing 'date' column"
        assert 'ticker' in df.columns, "Dataset missing 'ticker' column"
        assert 'returns' in df.columns, "Dataset missing 'returns' column"
        
        # Check for reasonable data ranges
        assert diagnostics['date_range']['unique_dates'] > 100, "Too few unique dates"
        assert diagnostics['tickers']['unique_tickers'] >= 25, "Too few tickers"
        
        # No major data quality issues
        assert diagnostics['duplicate_rows'] < 100, f"Too many duplicate rows: {diagnostics['duplicate_rows']}"
        
        print(f"✅ Dataset integrity check passed")
        print(f"   Shape: {diagnostics['shape']}")
        print(f"   Date range: {diagnostics['date_range']['start']} to {diagnostics['date_range']['end']}")
        print(f"   Tickers: {diagnostics['tickers']['unique_tickers']}")
        
        return diagnostics
    
    def test_technical_indicators_present(self, baseline_data_path):
        """Verify all expected technical indicators are present"""
        df = pd.read_parquet(baseline_data_path)
        
        # Expected indicators (based on your current implementation)
        expected_indicators = [
            'returns', 'log_returns', 'excess_returns', 'market_return',
            'volatility_5d', 'volatility_20d'
        ]
        
        # Check for normalized indicators
        normalized_indicators = [col for col in df.columns if col.endswith('_norm')]
        
        for indicator in expected_indicators:
            assert indicator in df.columns, f"Missing indicator: {indicator}"
        
        assert len(normalized_indicators) > 10, f"Too few normalized indicators: {len(normalized_indicators)}"
        
        print(f"✅ Technical indicators check passed")
        print(f"   Basic indicators: {len(expected_indicators)}")
        print(f"   Normalized indicators: {len(normalized_indicators)}")
        
        return {
            'basic_indicators': expected_indicators,
            'normalized_indicators': normalized_indicators
        }
    
    def test_data_pipeline_reproducibility(self, temp_data_dir):
        """Test that data pipeline produces consistent results"""
        print("🔄 Testing data pipeline reproducibility...")
        
        # Run data pipeline twice
        results = []
        
        for run_num in range(2):
            print(f"   Run {run_num + 1}/2...")
            
            # Set up isolated environment
            run_dir = Path(temp_data_dir) / f"run_{run_num}"
            run_dir.mkdir()
            
            # Change to run directory
            original_cwd = os.getcwd()
            os.chdir(run_dir)
            
            try:
                # Copy necessary files
                import shutil
                shutil.copytree(original_cwd / "varibad", run_dir / "varibad")
                
                # Run data pipeline
                result = subprocess.run([
                    sys.executable, str(original_cwd / "varibad" / "main.py"),
                    '--mode', 'data_only'
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    pytest.fail(f"Data pipeline failed on run {run_num}: {result.stderr}")
                
                # Load result
                output_file = run_dir / "data" / "sp500_rl_ready_cleaned.parquet"
                if output_file.exists():
                    df = pd.read_parquet(output_file)
                    results.append(df)
                else:
                    pytest.fail(f"Output file not created on run {run_num}")
                    
            finally:
                os.chdir(original_cwd)
        
        # Compare results
        assert len(results) == 2, "Did not get results from both runs"
        
        print("   Comparing results...")
        assert_dataframes_equal(results[0], results[1], tolerance=1e-10)
        
        print("✅ Data pipeline reproducibility test passed")


class TestTechnicalIndicatorValidation:
    """Test technical indicators against known values"""
    
    def test_simple_moving_average(self, sample_market_data):
        """Test SMA calculation against known values"""
        # Get sample data for one ticker
        ticker_data = sample_market_data[sample_market_data['ticker'] == 'AAPL'].copy()
        
        # Calculate SMA manually
        expected_sma_5 = ticker_data['close'].rolling(5).mean()
        
        # Compare with dataset
        actual_sma_5 = ticker_data['sma_5']
        
        # Should match (allowing for NaN handling)
        valid_indices = ~expected_sma_5.isna()
        np.testing.assert_array_almost_equal(
            actual_sma_5[valid_indices].values,
            expected_sma_5[valid_indices].values,
            decimal=6
        )
        
        print("✅ Simple Moving Average validation passed")
    
    def test_returns_calculation(self, sample_market_data):
        """Test returns calculation"""
        ticker_data = sample_market_data[sample_market_data['ticker'] == 'AAPL'].copy()
        
        # Calculate returns manually
        expected_returns = ticker_data['close'].pct_change()
        expected_log_returns = np.log(ticker_data['close'] / ticker_data['close'].shift(1))
        
        # Compare
        valid_indices = ~expected_returns.isna()
        
        np.testing.assert_array_almost_equal(
            ticker_data['returns'][valid_indices].values,
            expected_returns[valid_indices].values,
            decimal=10
        )
        
        np.testing.assert_array_almost_equal(
            ticker_data['log_returns'][valid_indices].values,
            expected_log_returns[valid_indices].values,
            decimal=10
        )
        
        print("✅ Returns calculation validation passed")
    
    def test_market_features(self, sample_market_data):
        """Test market-wide feature calculation"""
        # Market return should be average of individual returns
        market_data = sample_market_data.groupby('date').agg({
            'returns': 'mean',
            'market_return': 'first'
        }).reset_index()
        
        # Compare calculated vs stored market return
        valid_indices = ~market_data['returns'].isna()
        
        np.testing.assert_array_almost_equal(
            market_data['market_return'][valid_indices].values,
            market_data['returns'][valid_indices].values,
            decimal=10
        )
        
        print("✅ Market features validation passed")


class TestDataPipelinePerformance:
    """Test data pipeline performance characteristics"""
    
    def test_pipeline_execution_time(self, temp_data_dir):
        """Test that data pipeline completes within reasonable time"""
        print("⏱️  Testing data pipeline performance...")
        
        # Set up test environment
        test_dir = Path(temp_data_dir) / "performance_test"
        test_dir.mkdir()
        
        original_cwd = os.getcwd()
        os.chdir(test_dir)
        
        try:
            # Copy varibad directory
            import shutil
            shutil.copytree(original_cwd / "varibad", test_dir / "varibad")
            
            # Time the execution
            start_time = time.time()
            
            result = subprocess.run([
                sys.executable, str(original_cwd / "varibad" / "main.py"),
                '--mode', 'data_only'
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            execution_time = time.time() - start_time
            
            if result.returncode != 0:
                pytest.fail(f"Data pipeline failed: {result.stderr}")
            
            # Performance requirements
            assert execution_time < 300, f"Pipeline too slow: {execution_time:.1f}s > 300s"
            
            print(f"✅ Pipeline performance test passed: {execution_time:.1f}s")
            
            return execution_time
            
        finally:
            os.chdir(original_cwd)
    
    def test_memory_usage(self, baseline_data_path):
        """Test memory usage is reasonable"""
        # Load dataset and check memory usage
        df = pd.read_parquet(baseline_data_path)
        
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Should be reasonable for 30 stocks over 35 years
        assert memory_mb < 1000, f"Dataset uses too much memory: {memory_mb:.1f} MB"
        
        print(f"✅ Memory usage test passed: {memory_mb:.1f} MB")
        
        return memory_mb


class TestDataPipelineComponents:
    """Test individual components of the data pipeline"""
    
    def test_yfinance_download_simulation(self, test_config):
        """Test that we can simulate the yfinance download process"""
        # Create mock data similar to yfinance output
        tickers = ['AAPL', 'MSFT']
        
        mock_data = []
        for ticker in tickers:
            for i in range(10):
                date = pd.Timestamp('2020-01-01') + pd.Timedelta(days=i)
                price = 100 + i + np.random.normal(0, 1)
                
                mock_data.append({
                    'date': date,
                    'ticker': ticker,
                    'open': price,
                    'high': price * 1.02,
                    'low': price * 0.98,
                    'close': price,
                    'volume': 1000000
                })
        
        df = pd.DataFrame(mock_data)
        
        # Test that our data structure is compatible
        assert 'date' in df.columns
        assert 'ticker' in df.columns
        assert len(df) == 20  # 2 tickers * 10 days
        
        print("✅ Data structure compatibility test passed")
    
    def test_normalization_ranges(self, sample_market_data):
        """Test that normalization produces expected ranges"""
        # Check normalized columns
        norm_columns = [col for col in sample_market_data.columns if col.endswith('_norm')]
        
        for col in norm_columns:
            values = sample_market_data[col].dropna()
            if len(values) > 0:
                # Most normalized values should be reasonable
                assert values.min() >= -10, f"Column {col} has extreme negative values"
                assert values.max() <= 10, f"Column {col} has extreme positive values"
        
        print(f"✅ Normalization ranges test passed for {len(norm_columns)} columns")


# Performance regression test
def test_baseline_performance_comparison(baseline_capture, temp_data_dir):
    """Compare current performance against baseline"""
    # This test will be populated after we have a baseline
    pytest.skip("Baseline comparison requires captured baseline - run baseline_capture.py first")


# Integration test
def test_full_pipeline_integration(temp_data_dir):
    """Test that the complete pipeline works end-to-end"""
    print("🔗 Testing full pipeline integration...")
    
    test_dir = Path(temp_data_dir) / "integration_test"
    test_dir.mkdir()
    
    original_cwd = os.getcwd()
    os.chdir(test_dir)
    
    try:
        # Copy necessary files
        import shutil
        shutil.copytree(original_cwd / "varibad", test_dir / "varibad")
        
        # Run data processing
        result = subprocess.run([
            sys.executable, str(original_cwd / "varibad" / "main.py"),
            '--mode', 'data_only'
        ], capture_output=True, text=True, timeout=300)
        
        assert result.returncode == 0, f"Data processing failed: {result.stderr}"
        
        # Verify output exists
        output_file = test_dir / "data" / "sp500_rl_ready_cleaned.parquet"
        assert output_file.exists(), "Output dataset not created"
        
        # Load and verify
        df = pd.read_parquet(output_file)
        diagnostics = check_dataset_integrity(df)
        
        assert len(diagnostics['issues']) == 0, f"Data quality issues: {diagnostics['issues']}"
        
        print("✅ Full pipeline integration test passed")
        
    finally:
        os.chdir(original_cwd)