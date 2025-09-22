#!/usr/bin/env python3
"""
Debug script to test HMM pre-training step by step
"""

import sys
import logging
import traceback
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test all required imports"""
    try:
        logger.info("Testing imports...")
        
        # Test hmmlearn
        from hmmlearn.hmm import GaussianHMM
        logger.info("✅ hmmlearn import OK")
        
        # Test torch
        import torch
        logger.info("✅ torch import OK")
        
        # Test our modules
        from environments.data import PortfolioDataset
        logger.info("✅ PortfolioDataset import OK")
        
        from models.hmm_encoder import HMMEncoder  
        logger.info("✅ HMMEncoder import OK")
        
        from config import experiment_to_training_config, ExperimentConfig
        logger.info("✅ config imports OK")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_config():
    """Test config creation"""
    try:
        logger.info("Testing config creation...")
        
        from config import experiment_to_training_config, ExperimentConfig
        
        exp = ExperimentConfig(seed=0, asset_class="sp500", encoder="hmm")
        cfg = experiment_to_training_config(exp)
        
        logger.info(f"✅ Config created: asset_class={cfg.asset_class}, data_path={cfg.data_path}")
        
        # Check if data file exists
        data_path = Path(cfg.data_path)
        if data_path.exists():
            logger.info(f"✅ Dataset file exists: {data_path}")
        else:
            logger.error(f"❌ Dataset file missing: {data_path}")
            return False, None
            
        return True, cfg
        
    except Exception as e:
        logger.error(f"❌ Config creation failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False, None

def test_dataset_loading(cfg):
    """Test dataset loading"""
    try:
        logger.info("Testing dataset loading...")
        
        from environments.data import PortfolioDataset
        
        # Try to load dataset
        dataset = PortfolioDataset(
            asset_class=cfg.asset_class,
            data_path=cfg.data_path,
            split="train",
            train_end=cfg.train_end,
            val_end=cfg.val_end,
            proportional=getattr(cfg, 'proportional', False),
            proportions=getattr(cfg, 'proportions', (0.7, 0.2, 0.1))
        )
        
        logger.info(f"✅ Dataset loaded: {len(dataset.data)} rows, {dataset.num_assets} assets")
        
        # Test feature extraction
        features = dataset.data[dataset.feature_cols].values.reshape(
            len(dataset), dataset.num_assets, dataset.num_features
        )
        X = features.reshape(-1, dataset.num_features)
        
        logger.info(f"✅ Features extracted: {X.shape} (samples, features)")
        
        if X.shape[0] == 0:
            logger.error("❌ No training data found!")
            return False, None, None
            
        return True, dataset, X
        
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False, None, None

def test_hmm_fitting(X):
    """Test HMM fitting"""
    try:
        logger.info("Testing HMM fitting...")
        
        from hmmlearn.hmm import GaussianHMM
        import numpy as np
        
        logger.info(f"Data shape: {X.shape}")
        logger.info(f"Data stats: min={X.min():.4f}, max={X.max():.4f}, mean={X.mean():.4f}")
        
        # Check for NaN/inf
        if np.isnan(X).any():
            logger.error("❌ Data contains NaN values!")
            return False
            
        if np.isinf(X).any():
            logger.error("❌ Data contains infinite values!")
            return False
        
        # Fit HMM
        hmm = GaussianHMM(
            n_components=4, 
            covariance_type="full", 
            n_iter=10,  # Reduce iterations for quick test
            random_state=0,
            tol=1e-3
        )
        
        logger.info("Fitting HMM...")
        hmm.fit(X)
        
        converged = hmm.monitor_.converged
        log_likelihood = hmm.score(X)
        
        logger.info(f"✅ HMM fit complete: converged={converged}, loglik={log_likelihood:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ HMM fitting failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all tests"""
    logger.info("🔍 Starting HMM pre-training debug...")
    
    # Test 1: Imports
    if not test_imports():
        logger.error("❌ Import test failed")
        return False
    
    # Test 2: Config
    success, cfg = test_config()
    if not success:
        logger.error("❌ Config test failed")
        return False
    
    # Test 3: Dataset
    success, dataset, X = test_dataset_loading(cfg)
    if not success:
        logger.error("❌ Dataset test failed")
        return False
    
    # Test 4: HMM fitting
    if not test_hmm_fitting(X):
        logger.error("❌ HMM fitting test failed")
        return False
    
    logger.info("🎉 All tests passed! HMM pre-training should work.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)