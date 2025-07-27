#!/usr/bin/env python3
"""
Complete system test for restructured VariBAD repository
Tests all components and creates a minimal experiment to verify everything works

Run this once after restructuring, then delete it.
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        # Test core imports
        import torch
        import numpy as np
        import pandas as pd
        import yfinance as yf
        import gym
        
        logger.info("✓ Core dependencies imported successfully")
        
        # Test project imports
        from varibad.data import load_dataset, create_dataset
        from varibad.models import VariBADVAE, PortfolioEnvironment
        from varibad.trainer import VariBADTrainer
        from varibad.utils import TrajectoryBuffer, get_device
        
        logger.info("✓ VariBAD modules imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_device_detection():
    """Test device detection"""
    logger.info("Testing device detection...")
    
    try:
        from varibad.utils import get_device
        
        device = get_device('auto')
        logger.info(f"✓ Device detected: {device}")
        
        # Test PyTorch CUDA availability
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("ℹ CUDA not available, using CPU")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Device detection failed: {e}")
        return False


def test_data_creation():
    """Test data creation (with minimal data)"""
    logger.info("Testing data creation...")
    
    try:
        from varibad.data import download_stock_data, add_technical_indicators, normalize_features, clean_data
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            test_data_path = Path(temp_dir) / "test_data.parquet"
            
            # Download minimal data (just 2 tickers, 1 year)
            logger.info("Downloading minimal test data...")
            tickers = ['AAPL', 'MSFT']  # Just 2 stocks
            raw_data = download_stock_data(
                tickers=tickers, 
                start_date='2024-01-01', 
                end_date='2024-12-31'
            )
            
            logger.info(f"✓ Downloaded data: {raw_data.shape}")
            
            # Process data
            with_indicators = add_technical_indicators(raw_data)
            logger.info(f"✓ Added indicators: {with_indicators.shape}")
            
            normalized = normalize_features(with_indicators)
            logger.info(f"✓ Normalized features: {normalized.shape}")
            
            cleaned = clean_data(normalized)
            logger.info(f"✓ Cleaned data: {cleaned.shape}")
            
            # Save test data
            cleaned.to_parquet(test_data_path)
            logger.info(f"✓ Saved test data: {test_data_path}")
            
            return True
            
    except Exception as e:
        logger.error(f"✗ Data creation failed: {e}")
        return False


def test_model_creation():
    """Test model instantiation"""
    logger.info("Testing model creation...")
    
    try:
        from varibad.models import VariBADVAE, PortfolioEnvironment
        import torch
        import pandas as pd
        import numpy as np
        
        # Create minimal test data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        tickers = ['AAPL', 'MSFT']
        
        test_data = []
        for date in dates:
            for ticker in tickers:
                test_data.append({
                    'date': date,
                    'ticker': ticker,
                    'returns': np.random.normal(0.001, 0.02),
                    'market_return': np.random.normal(0.001, 0.015),
                    'excess_returns': np.random.normal(0, 0.01),
                    'volatility_5d': np.random.uniform(0.01, 0.05),
                    'volatility_20d': np.random.uniform(0.01, 0.04),
                    'close_norm': np.random.uniform(0, 1),
                    'rsi_norm': np.random.uniform(-1, 1),
                    'volume_norm': np.random.normal(0, 1)
                })
        
        df = pd.DataFrame(test_data)
        
        # Test environment
        env = PortfolioEnvironment(
            data=df,
            episode_length=10,
            enable_short_selling=True
        )
        
        logger.info(f"✓ Environment created: {env.n_assets} assets")
        
        # Test environment reset
        state = env.reset()
        logger.info(f"✓ Environment reset: state shape {state.shape}")
        
        # Test model creation
        varibad = VariBADVAE(
            state_dim=len(state),
            action_dim=env.action_space.shape[0],
            latent_dim=3
        )
        
        param_count = sum(p.numel() for p in varibad.parameters())
        logger.info(f"✓ VariBAD model created: {param_count:,} parameters")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model creation failed: {e}")
        return False


def test_training_loop():
    """Test minimal training loop"""
    logger.info("Testing training loop...")
    
    try:
        from varibad.trainer import VariBADTrainer
        
        # Create minimal config
        test_config = {
            "experiment_name": "system_test",
            "training": {
                "num_iterations": 3,  # Very short
                "episode_length": 5,
                "episodes_per_iteration": 2,
                "vae_updates": 2,
                "buffer_size": 10
            },
            "model": {
                "latent_dim": 3,
                "encoder_hidden": 32,
                "decoder_hidden": 32,
                "policy_hidden": 64
            },
            "portfolio": {
                "short_selling": False,  # Simpler
                "transaction_cost": 0.0
            },
            "learning_rates": {
                "policy_lr": 0.01,
                "vae_encoder_lr": 0.01,
                "vae_decoder_lr": 0.01
            },
            "environment": {
                "device": "cpu"  # Force CPU for testing
            }
        }
        
        # Create minimal test data file
        test_data_dir = Path("test_data")
        test_data_dir.mkdir(exist_ok=True)
        test_data_path = test_data_dir / "test_dataset.parquet"
        
        # Generate minimal synthetic data
        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        
        test_data = []
        for date in dates:
            for ticker in tickers:
                test_data.append({
                    'date': date,
                    'ticker': ticker,
                    'returns': np.random.normal(0.001, 0.02),
                    'market_return': np.random.normal(0.001, 0.015),
                    'excess_returns': np.random.normal(0, 0.01),
                    'volatility_5d': np.random.uniform(0.01, 0.05),
                    'volatility_20d': np.random.uniform(0.01, 0.04),
                    'close_norm': np.random.uniform(0, 1),
                    'rsi_norm': np.random.uniform(-1, 1),
                    'volume_norm': np.random.normal(0, 1),
                    'open_norm': np.random.uniform(0, 1),
                    'high_norm': np.random.uniform(0, 1),
                    'low_norm': np.random.uniform(0, 1),
                })
        
        df = pd.DataFrame(test_data)
        df.to_parquet(test_data_path)
        
        # Update config to use test data
        test_config["environment"]["data_path"] = str(test_data_path)
        
        # Initialize trainer
        trainer = VariBADTrainer(test_config)
        logger.info("✓ Trainer initialized")
        
        # Run minimal training
        stats = trainer.train()
        logger.info("✓ Training completed")
        logger.info(f"✓ Collected {len(stats.get('iteration', []))} iterations of stats")
        
        # Test checkpoint saving
        checkpoint_path = test_data_dir / "test_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))
        logger.info("✓ Checkpoint saved")
        
        # Clean up
        shutil.rmtree(test_data_dir)
        logger.info("✓ Test data cleaned up")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Training loop failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_experiment_runner():
    """Test the main experiment runner"""
    logger.info("Testing experiment runner...")
    
    try:
        # Create minimal test config
        test_config_dir = Path("test_config")
        test_config_dir.mkdir(exist_ok=True)
        
        test_config = {
            "experiment_name": "runner_test",
            "training": {
                "num_iterations": 2,
                "episode_length": 5,
                "episodes_per_iteration": 1,
                "vae_updates": 1
            },
            "model": {
                "latent_dim": 2,
                "encoder_hidden": 16,
                "decoder_hidden": 16,
                "policy_hidden": 32
            },
            "portfolio": {
                "short_selling": False
            },
            "environment": {
                "device": "cpu"
            }
        }
        
        config_path = test_config_dir / "test_experiment.conf"
        with open(config_path, 'w') as f:
            json.dump(test_config, f, indent=2)
        
        # Create minimal test data
        test_data_path = test_config_dir / "minimal_data.parquet"
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        tickers = ['AAPL', 'MSFT']
        
        test_data = []
        for date in dates:
            for ticker in tickers:
                test_data.append({
                    'date': date,
                    'ticker': ticker,
                    'returns': np.random.normal(0.001, 0.02),
                    'market_return': np.random.normal(0.001, 0.015),
                    'excess_returns': np.random.normal(0, 0.01),
                    'volatility_5d': 0.02,
                    'volatility_20d': 0.025,
                    'close_norm': 0.5,
                    'rsi_norm': 0.0,
                    'volume_norm': 0.0,
                    'open_norm': 0.5,
                    'high_norm': 0.55,
                    'low_norm': 0.45,
                })
        
        df = pd.DataFrame(test_data)
        df.to_parquet(test_data_path)
        
        # Update config
        test_config["environment"]["data_path"] = str(test_data_path)
        with open(config_path, 'w') as f:
            json.dump(test_config, f, indent=2)
        
        # Test experiment runner
        logger.info("Running test experiment...")
        
        # Import and run
        sys.path.append('.')
        from run_experiment import run_experiment
        
        result_path = run_experiment(str(config_path))
        
        logger.info(f"✓ Experiment completed: {result_path}")
        
        # Verify zip file exists
        if Path(result_path).exists():
            logger.info("✓ Result zip file created successfully")
        else:
            logger.error("✗ Result zip file not found")
            return False
        
        # Clean up
        shutil.rmtree(test_config_dir)
        
        # Clean up any results
        results_dir = Path("results")
        if results_dir.exists():
            shutil.rmtree(results_dir)
        
        logger.info("✓ Test files cleaned up")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Experiment runner failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_all_tests():
    """Run all system tests"""
    logger.info("🧪 Running VariBAD System Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Device Detection", test_device_detection),
        ("Data Creation", test_data_creation),
        ("Model Creation", test_model_creation),
        ("Training Loop", test_training_loop),
        ("Experiment Runner", test_experiment_runner),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            success = test_func()
            results[test_name] = success
            
            if success:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} CRASHED: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("🎯 TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name:.<30} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! System is ready for use.")
        logger.info("\nYou can now:")
        logger.info("1. Delete this test file: rm test_setup.py")
        logger.info("2. Run your first experiment: python run_experiment.py config/experiment1.conf")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        logger.error("The system may not work correctly until these issues are resolved.")
    
    return passed == total


if __name__ == "__main__":
    # Import required modules for testing
    import pandas as pd
    import numpy as np
    
    success = run_all_tests()
    sys.exit(0 if success else 1)