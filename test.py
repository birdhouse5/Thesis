# test_splits.py - Comprehensive test suite for train-test-val splits
import pytest
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import tempfile
import os
from datetime import datetime

# Import your modules
from environments.dataset import Dataset, create_split_datasets
from environments.env import MetaEnv
from models.policy import PortfolioPolicy
from models.vae import VAE

class TestSplits:
    @pytest.fixture
    def mock_dataset_path(self):
        """Create a mock dataset spanning 2014-2022 for testing splits"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = f.name
        
        # Generate mock data spanning split boundaries
        dates = pd.date_range('2014-01-01', '2022-12-31', freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL']  # 3 assets for testing
        
        data = []
        for date in dates:
            for ticker in tickers:
                row = {
                    'date': date,
                    'ticker': ticker,
                    'close': np.random.uniform(50, 200),
                    'returns': np.random.randn() * 0.02,
                    # Add normalized features (required for Dataset)
                    'close_norm': np.random.uniform(0, 1),
                    'returns_norm': np.random.randn(),
                    'rsi_norm': np.random.uniform(-1, 1),
                    'volume_norm': np.random.randn(),
                    'volatility_5d_norm': np.random.randn(),
                }
                data.append(row)
        
        df = pd.DataFrame(data)
        df.to_parquet(temp_path, index=False)
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_split_date_boundaries(self, mock_dataset_path):
        """Test that splits respect date boundaries with no overlap"""
        datasets = create_split_datasets(mock_dataset_path)
        
        # Get date ranges for each split
        train_dates = datasets['train'].data['date']
        val_dates = datasets['val'].data['date']
        test_dates = datasets['test'].data['date']
        
        # Test boundaries
        assert train_dates.max() <= pd.to_datetime('2015-12-31'), "Train data leaks into validation"
        assert val_dates.min() > pd.to_datetime('2015-12-31'), "Validation starts before train ends"
        assert val_dates.max() <= pd.to_datetime('2020-12-31'), "Validation data leaks into test"
        assert test_dates.min() > pd.to_datetime('2020-12-31'), "Test starts before validation ends"
        
        # Test no overlap
        assert len(set(train_dates) & set(val_dates)) == 0, "Train-val overlap found"
        assert len(set(train_dates) & set(test_dates)) == 0, "Train-test overlap found"
        assert len(set(val_dates) & set(test_dates)) == 0, "Val-test overlap found"
    
    def test_split_consistency(self, mock_dataset_path):
        """Test that all splits have consistent structure"""
        datasets = create_split_datasets(mock_dataset_path)
        
        # All splits should have same tickers and features
        train_tickers = set(datasets['train'].tickers)
        val_tickers = set(datasets['val'].tickers)
        test_tickers = set(datasets['test'].tickers)
        
        assert train_tickers == val_tickers == test_tickers, "Inconsistent tickers across splits"
        
        # Same number of assets and features
        assert datasets['train'].num_assets == datasets['val'].num_assets == datasets['test'].num_assets
        assert datasets['train'].num_features == datasets['val'].num_features == datasets['test'].num_features
        
        # Same feature columns
        assert datasets['train'].feature_cols == datasets['val'].feature_cols == datasets['test'].feature_cols
    
    def test_rectangular_structure(self, mock_dataset_path):
        """Test that each split maintains rectangular structure"""
        datasets = create_split_datasets(mock_dataset_path)
        
        for split_name, dataset in datasets.items():
            expected_rows = dataset.num_days * dataset.num_assets
            actual_rows = len(dataset.data)
            assert actual_rows == expected_rows, f"{split_name} split not rectangular: {actual_rows} != {expected_rows}"
    
    def test_environment_creation(self, mock_dataset_path):
        """Test that environments can be created from all splits"""
        datasets = create_split_datasets(mock_dataset_path)
        
        for split_name, dataset in datasets.items():
            # Create dataset tensor
            window_size = min(60, len(dataset))
            window = dataset.get_window(0, window_size)
            
            dataset_tensor = {
                'features': torch.tensor(window['features'], dtype=torch.float32),
                'raw_prices': torch.tensor(window['raw_prices'], dtype=torch.float32)
            }
            
            # Create environment
            env = MetaEnv(
                dataset=dataset_tensor,
                feature_columns=dataset.feature_cols,
                seq_len=min(30, window_size),
                min_horizon=5,
                max_horizon=10
            )
            
            # Test basic functionality
            task = env.sample_task()
            env.set_task(task)
            obs = env.reset()
            
            assert obs.shape == (dataset.num_assets, dataset.num_features), f"{split_name} env obs shape mismatch"
    
    def test_model_compatibility(self, mock_dataset_path):
        """Test that models can be initialized with split data"""
        datasets = create_split_datasets(mock_dataset_path)
        dataset = datasets['train']  # Use train split for model init
        
        # Get observation shape
        window = dataset.get_window(0, min(30, len(dataset)))
        obs_shape = (dataset.num_assets, dataset.num_features)
        
        # Test VAE initialization
        vae = VAE(
            obs_dim=obs_shape,
            num_assets=dataset.num_assets,
            latent_dim=16,
            hidden_dim=64
        )
        
        # Test Policy initialization
        policy = PortfolioPolicy(
            obs_shape=obs_shape,
            latent_dim=16,
            num_assets=dataset.num_assets,
            hidden_dim=64
        )
        
        # Test forward pass
        batch_size = 2
        mock_obs = torch.randn(batch_size, *obs_shape)
        mock_latent = torch.randn(batch_size, 16)
        
        action, value = policy.act(mock_obs, mock_latent, deterministic=True)
        assert action.shape == (batch_size, dataset.num_assets), "Policy output shape mismatch"
        assert torch.allclose(action.sum(dim=-1), torch.ones(batch_size), atol=1e-5), "Portfolio weights don't sum to 1"
    
    def test_empty_split_handling(self, mock_dataset_path):
        """Test handling of edge cases like empty splits"""
        # Test with split dates that create empty validation split
        try:
            datasets = create_split_datasets(
                mock_dataset_path,
                train_end='2013-12-31',  # Before our data starts
                val_end='2014-06-30'
            )
            assert False, "Should have raised error for empty train split"
        except ValueError as e:
            assert "No data found" in str(e), "Wrong error message for empty split"
    
    def test_tensor_shapes_pipeline(self, mock_dataset_path):
        """Test tensor shapes throughout the full pipeline"""
        datasets = create_split_datasets(mock_dataset_path)
        
        for split_name, dataset in datasets.items():
            # Create environment
            window_size = min(40, len(dataset))
            window = dataset.get_window(0, window_size)
            
            dataset_tensor = {
                'features': torch.tensor(window['features'], dtype=torch.float32),
                'raw_prices': torch.tensor(window['raw_prices'], dtype=torch.float32)
            }
            
            env = MetaEnv(dataset=dataset_tensor, feature_columns=dataset.feature_cols,
                         seq_len=20, min_horizon=5, max_horizon=10)
            
            # Sample task and reset
            task = env.sample_task()
            env.set_task(task)
            obs = env.reset()
            
            # Verify shapes
            expected_obs_shape = (dataset.num_assets, dataset.num_features)
            assert obs.shape == expected_obs_shape, f"{split_name}: obs shape {obs.shape} != {expected_obs_shape}"
            
            # Test environment step
            action = np.random.rand(dataset.num_assets)
            action = action / action.sum()  # Normalize
            
            next_obs, reward, done, info = env.step(action)
            assert isinstance(reward, (int, float)), f"{split_name}: reward should be scalar"
            assert isinstance(done, bool), f"{split_name}: done should be boolean"
    
    def test_split_info_consistency(self, mock_dataset_path):
        """Test that split info is correctly reported"""
        datasets = create_split_datasets(mock_dataset_path)
        
        total_days = 0
        for split_name, dataset in datasets.items():
            info = dataset.get_split_info()
            
            # Basic info checks
            assert info['split'] == split_name
            assert info['num_days'] > 0, f"{split_name} has 0 days"
            assert info['num_assets'] > 0, f"{split_name} has 0 assets"
            assert info['num_features'] > 0, f"{split_name} has 0 features"
            
            total_days += info['num_days']
        
        # Total days should be reasonable (not testing exact number due to weekends/holidays)
        assert total_days > 1000, "Total days across splits seems too low"


def run_tests():
    """Run all tests manually (for environments without pytest)"""
    import tempfile
    
    print("🧪 Running Split Testing Suite...")
    
    # Create test instance
    test_instance = TestSplits()
    
    # Create mock dataset
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        temp_path = f.name
    
    # Generate mock data
    dates = pd.date_range('2014-01-01', '2022-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    data = []
    for date in dates:
        for ticker in tickers:
            row = {
                'date': date,
                'ticker': ticker,
                'close': np.random.uniform(50, 200),
                'returns': np.random.randn() * 0.02,
                'close_norm': np.random.uniform(0, 1),
                'returns_norm': np.random.randn(),
                'rsi_norm': np.random.uniform(-1, 1),
                'volume_norm': np.random.randn(),
                'volatility_5d_norm': np.random.randn(),
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    df.to_parquet(temp_path, index=False)
    
    try:
        # Run tests
        tests = [
            ('Split Date Boundaries', test_instance.test_split_date_boundaries),
            ('Split Consistency', test_instance.test_split_consistency),
            ('Rectangular Structure', test_instance.test_rectangular_structure),
            ('Environment Creation', test_instance.test_environment_creation),
            ('Model Compatibility', test_instance.test_model_compatibility),
            ('Empty Split Handling', test_instance.test_empty_split_handling),
            ('Tensor Shapes Pipeline', test_instance.test_tensor_shapes_pipeline),
            ('Split Info Consistency', test_instance.test_split_info_consistency),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                print(f"  ✓ {test_name}...", end=' ')
                test_func(temp_path)
                print("PASSED")
                passed += 1
            except Exception as e:
                print(f"FAILED: {e}")
                failed += 1
        
        print(f"\n📊 Results: {passed} passed, {failed} failed")
        
        if failed == 0:
            print("🎉 All tests passed! Your split implementation is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the errors above.")
            
    finally:
        # Cleanup
        os.unlink(temp_path)


if __name__ == "__main__":
    # Run tests if executed directly
    run_tests()