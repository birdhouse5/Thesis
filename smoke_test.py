#!/usr/bin/env python3
"""
Comprehensive Smoke Test for Portfolio Optimization Pipeline

This script runs minimal versions of all pipeline components to validate:
- Dataset creation and loading for both SP500 and crypto
- Environment initialization and basic functionality
- Model creation and forward passes for all encoder types
- Training loop execution (minimal episodes)
- Evaluation and backtesting functionality
- MLflow integration and artifact logging

Usage:
    python smoke_test.py [--quick] [--no-mlflow] [--verbose]
"""

import os
import sys
import tempfile
import shutil
import traceback
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings

import torch
import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup logging
def setup_test_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

# Test configurations
TEST_CONFIGS = {
    "quick": {
        "max_episodes": 5,
        "val_episodes": 2,
        "test_episodes": 3,
        "val_interval": 2,
        "seq_len": 5,
        "min_horizon": 2,
        "max_horizon": 4,
        "batch_size": 32
    },
    "normal": {
        "max_episodes": 20,
        "val_episodes": 5,
        "test_episodes": 10,
        "val_interval": 5,
        "seq_len": 60,
        "min_horizon": 20,
        "max_horizon": 30,
        "batch_size": 128
    }
}

class SmokeTestResult:
    """Container for smoke test results"""
    def __init__(self):
        self.passed_tests: List[str] = []
        self.failed_tests: List[str] = []
        self.errors: Dict[str, str] = {}
        self.timings: Dict[str, float] = {}
        self.start_time = time.time()
    
    def add_pass(self, test_name: str, duration: float):
        self.passed_tests.append(test_name)
        self.timings[test_name] = duration
        logger.info(f"✅ {test_name} PASSED ({duration:.2f}s)")
    
    def add_fail(self, test_name: str, error: str, duration: float):
        self.failed_tests.append(test_name)
        self.errors[test_name] = error
        self.timings[test_name] = duration
        logger.error(f"❌ {test_name} FAILED ({duration:.2f}s): {error}")
    
    def print_summary(self):
        total_time = time.time() - self.start_time
        total_tests = len(self.passed_tests) + len(self.failed_tests)
        
        print(f"\n{'='*60}")
        print("SMOKE TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests: {total_tests}")
        print(f"Passed: {len(self.passed_tests)}")
        print(f"Failed: {len(self.failed_tests)}")
        print(f"Success rate: {len(self.passed_tests)/total_tests*100:.1f}%")
        print(f"Total time: {total_time:.1f}s")
        
        if self.failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for test in self.failed_tests:
                print(f"  - {test}: {self.errors[test][:100]}...")
        
        print(f"\n⏱️  TIMING BREAKDOWN:")
        for test, duration in sorted(self.timings.items(), key=lambda x: x[1], reverse=True):
            status = "✅" if test in self.passed_tests else "❌"
            print(f"  {status} {test}: {duration:.2f}s")
        
        return len(self.failed_tests) == 0

class SmokeTest:
    """Main smoke test runner"""
    
    def __init__(self, quick_mode: bool = False, enable_mlflow: bool = True, verbose: bool = False):
        self.quick_mode = quick_mode
        self.enable_mlflow = enable_mlflow
        self.verbose = verbose
        self.config = TEST_CONFIGS["quick" if quick_mode else "normal"]
        self.temp_dir = None
        self.original_cwd = os.getcwd()
        self.results = SmokeTestResult()
        
        # Initialize temp directory for test data
        self.temp_dir = tempfile.mkdtemp(prefix="portfolio_smoke_test_")
        
        global logger
        logger = logging.getLogger(__name__)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def cleanup(self):
        """Clean up temporary files and reset environment"""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
        os.chdir(self.original_cwd)
    
    def run_test(self, test_name: str, test_func):
        """Run a single test with error handling and timing"""
        start_time = time.time()
        try:
            logger.info(f"🧪 Running {test_name}...")
            test_func()
            duration = time.time() - start_time
            self.results.add_pass(test_name, duration)
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            if self.verbose:
                error_msg += f"\n{traceback.format_exc()}"
            self.results.add_fail(test_name, error_msg, duration)
    
    def test_imports(self):
        """Test that all required modules can be imported"""
        logger.info("Testing imports...")
        
        # Core modules
        from config import generate_experiment_configs, experiment_to_training_config
        from environments.data_preparation import create_dataset, create_crypto_dataset
        from environments.dataset import create_split_datasets
        from environments.env import MetaEnv
        from models.policy import PortfolioPolicy
        from models.vae import VAE
        from models.hmm_encoder import HMMEncoder
        from algorithms.trainer import PPOTrainer
        from evaluation_backtest import evaluate, run_sequential_backtest
        
        if self.enable_mlflow:
            from mlflow_logger import setup_mlflow
        
        logger.info("All imports successful")
    
    def test_sp500_dataset_creation(self):
        """Test SP500 dataset creation with full dataset"""
        logger.info("Testing SP500 dataset creation...")
        
        from environments.data_preparation import create_dataset
        
        # Create full SP500 dataset
        sp500_path = Path(self.temp_dir) / "test_sp500.parquet"
        
        # Use full dataset configuration - hardcoded values, no config references
        created_path = create_dataset(
            output_path=str(sp500_path),
            tickers=None,  # Use all SP500_TICKERS from the module
            start_date='1990-01-01',  # Full date range
            end_date='2025-01-01',
            force_recreate=True
        )
        
        # Verify dataset
        df = pd.read_parquet(created_path)
        assert len(df) > 0, f"Dataset is empty - got {len(df)} rows"
        assert 'ticker' in df.columns, "Missing ticker column"
        assert 'returns' in df.columns, "Missing returns column"
        
        # Should have substantial data - realistic expectations
        assert len(df) > 5000, f"Dataset too small: {len(df)} rows"
        assert df['ticker'].nunique() >= 20, f"Too few tickers: {df['ticker'].nunique()}"
        
        logger.info(f"SP500 dataset created: {df.shape} rows, {df['ticker'].nunique()} tickers")

    def test_crypto_dataset_creation(self):
        """Test crypto dataset creation with full dataset"""
        logger.info("Testing crypto dataset creation...")
        
        from environments.data_preparation import create_crypto_dataset
        
        # Create full crypto dataset
        crypto_path = Path(self.temp_dir) / "test_crypto.parquet"
        
        # Use full dataset configuration - hardcoded values, no config references
        created_path = create_crypto_dataset(
            output_path=str(crypto_path),
            tickers=None,  # Use all CRYPTO_TICKERS from the module
            days=92,  # Standard 92 days
            force_recreate=True
        )
        
        # Verify dataset
        df = pd.read_parquet(created_path)
        assert len(df) > 0, f"Dataset is empty - got {len(df)} rows"
        assert 'ticker' in df.columns, "Missing ticker column"
        assert 'returns' in df.columns, "Missing returns column"
        
        # Should have substantial data - realistic expectations
        assert len(df) > 50000, f"Dataset too small: {len(df)} rows"
        assert df['ticker'].nunique() >= 20, f"Too few tickers: {df['ticker'].nunique()}"
        
        logger.info(f"Crypto dataset created: {df.shape} rows, {df['ticker'].nunique()} tickers")
    
    def test_dataset_loading_and_splits(self):
        """Test dataset loading and temporal splitting"""
        logger.info("Testing dataset loading and splits...")
        
        from environments.dataset import create_split_datasets
        
        # Test with SP500 dataset
        sp500_path = Path(self.temp_dir) / "test_sp500.parquet"
        
        datasets = create_split_datasets(
            data_path=str(sp500_path),
            train_end='2020-01-15',
            val_end='2020-01-25',
            proportional=False
        )
        
        # Verify splits
        assert 'train' in datasets, "Missing train split"
        assert 'val' in datasets, "Missing val split"
        assert 'test' in datasets, "Missing test split"
        
        for split_name, dataset in datasets.items():
            assert len(dataset) > 0, f"{split_name} split is empty"
            assert dataset.num_assets > 0, f"{split_name} has no assets"
            assert dataset.num_features > 0, f"{split_name} has no features"
        
        logger.info("Dataset splits created successfully")
    
    def test_environment_creation(self):
        """Test MetaEnv creation and basic functionality"""
        logger.info("Testing environment creation...")
        
        from environments.env import MetaEnv
        from environments.dataset import create_split_datasets
        
        # Load dataset
        sp500_path = Path(self.temp_dir) / "test_sp500.parquet"
        datasets = create_split_datasets(str(sp500_path))
        train_dataset = datasets['train']
        
        # Create tensor data for environment
        seq_len_safe = min(self.config["seq_len"], len(train_dataset) - 1)
        window = train_dataset.get_window(0, seq_len_safe)
        
        mock_dataset = {
            'features': torch.tensor(window['features'], dtype=torch.float32),
            'raw_prices': torch.tensor(window['raw_prices'], dtype=torch.float32)
        }
        
        # Create environment
        env = MetaEnv(
            dataset=mock_dataset,
            feature_columns=train_dataset.feature_cols,
            seq_len=seq_len_safe,
            min_horizon=self.config["min_horizon"],
            max_horizon=self.config["max_horizon"],
            eta=0.05,  # Fixed: use proper eta value, not max_horizon
            rf_rate=0.02,
            transaction_cost_rate=0.001
        )
        
        # Test basic environment functionality
        task = env.sample_task()
        env.set_task(task)
        obs = env.reset()
        
        assert obs is not None, "Environment reset failed"
        assert obs.shape[0] == train_dataset.num_assets, "Wrong observation shape"
        
        # Test environment step
        action = np.random.rand(train_dataset.num_assets) * 0.1  # Small random weights
        next_obs, reward, done, info = env.step(action)
        
        assert isinstance(reward, (int, float)), "Reward should be scalar"
        assert isinstance(done, bool), "Done should be boolean"
        assert 'weights' in info, "Info should contain weights"
        
        logger.info("Environment creation and basic functionality tested")
    
    def test_model_creation_vae(self):
        """Test VAE model creation and forward pass"""
        logger.info("Testing VAE model creation...")
        
        from models.vae import VAE
        from environments.dataset import create_split_datasets
        
        # Get observation dimensions
        sp500_path = Path(self.temp_dir) / "test_sp500.parquet"
        datasets = create_split_datasets(str(sp500_path))
        train_dataset = datasets['train']
        
        obs_shape = (train_dataset.num_assets, train_dataset.num_features)
        
        # Create VAE
        vae = VAE(
            obs_dim=obs_shape,
            num_assets=train_dataset.num_assets,
            latent_dim=64,
            hidden_dim=256
        )
        
        # Test forward pass
        batch_size = 2
        seq_len = 5
        
        obs_seq = torch.randn(batch_size, seq_len, *obs_shape)
        action_seq = torch.randn(batch_size, seq_len, train_dataset.num_assets)
        reward_seq = torch.randn(batch_size, seq_len, 1)
        
        # Test encoding
        mu, logvar, hidden = vae.encode(obs_seq, action_seq, reward_seq)
        assert mu.shape == (batch_size, 64), f"Wrong mu shape: {mu.shape}"
        assert logvar.shape == (batch_size, 64), f"Wrong logvar shape: {logvar.shape}"
        
        # Test loss computation
        loss, components = vae.compute_loss(obs_seq, action_seq, reward_seq)
        assert isinstance(loss, torch.Tensor), "Loss should be tensor"
        assert loss.item() >= 0, "Loss should be non-negative"
        
        logger.info("VAE model creation and forward pass tested")
    
    def test_model_creation_hmm(self):
        """Test HMM encoder creation and forward pass"""
        logger.info("Testing HMM encoder creation...")
        
        from models.hmm_encoder import HMMEncoder
        from environments.dataset import create_split_datasets
        
        # Get observation dimensions
        sp500_path = Path(self.temp_dir) / "test_sp500.parquet"
        datasets = create_split_datasets(str(sp500_path))
        train_dataset = datasets['train']
        
        obs_shape = (train_dataset.num_assets, train_dataset.num_features)
        
        # Create HMM encoder
        hmm_encoder = HMMEncoder(
            obs_dim=obs_shape,
            num_assets=train_dataset.num_assets,
            latent_dim=4,  # 4 regimes
            hidden_dim=256
        )
        
        # Test forward pass
        batch_size = 2
        seq_len = 5
        
        obs_seq = torch.randn(batch_size, seq_len, *obs_shape)
        action_seq = torch.randn(batch_size, seq_len, train_dataset.num_assets)
        reward_seq = torch.randn(batch_size, seq_len, 1)
        
        # Test encoding
        regime_probs = hmm_encoder.encode(obs_seq, action_seq, reward_seq)
        assert regime_probs.shape == (batch_size, 4), f"Wrong regime_probs shape: {regime_probs.shape}"
        
        # Check probabilities sum to 1
        prob_sums = regime_probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6), "Probabilities don't sum to 1"
        
        logger.info("HMM encoder creation and forward pass tested")
    
    def test_model_creation_policy(self):
        """Test policy model creation and forward pass"""
        logger.info("Testing policy model creation...")
        
        from models.policy import PortfolioPolicy
        from environments.dataset import create_split_datasets
        
        # Get observation dimensions
        sp500_path = Path(self.temp_dir) / "test_sp500.parquet"
        datasets = create_split_datasets(str(sp500_path))
        train_dataset = datasets['train']
        
        obs_shape = (train_dataset.num_assets, train_dataset.num_features)
        
        # Create policy
        policy = PortfolioPolicy(
            obs_shape=obs_shape,
            latent_dim=64,
            num_assets=train_dataset.num_assets,
            hidden_dim=256
        )
        
        # Test forward pass
        batch_size = 2
        obs = torch.randn(batch_size, *obs_shape)
        latent = torch.randn(batch_size, 64)
        
        output = policy.forward(obs, latent)
        assert 'raw_actions' in output, "Missing raw_actions in output"
        assert 'value' in output, "Missing value in output"
        assert output['raw_actions'].shape == (batch_size, train_dataset.num_assets), "Wrong action shape"
        assert output['value'].shape == (batch_size, 1), "Wrong value shape"
        
        # Test action sampling
        actions, values = policy.act(obs, latent, deterministic=True)
        assert actions.shape == (batch_size, train_dataset.num_assets), "Wrong action shape"
        
        logger.info("Policy model creation and forward pass tested")
    
    def test_training_loop_vae(self):
        """Test minimal training loop with VAE encoder"""
        logger.info("Testing training loop with VAE...")
        
        self._test_training_loop("vae")
    
    def test_training_loop_none(self):
        """Test minimal training loop with no encoder"""
        logger.info("Testing training loop with no encoder...")
        
        self._test_training_loop("none")
    
    def test_training_loop_hmm(self):
        """Test minimal training loop with HMM encoder"""
        logger.info("Testing training loop with HMM...")
        
        self._test_training_loop("hmm")
    
    def _test_training_loop(self, encoder_type: str):
        """Generic training loop test"""
        from config import ExperimentConfig, experiment_to_training_config
        from environments.dataset import create_split_datasets
        from environments.env import MetaEnv
        from models.policy import PortfolioPolicy
        from models.vae import VAE
        from models.hmm_encoder import HMMEncoder
        from algorithms.trainer import PPOTrainer
        
        # Create test config
        exp_config = ExperimentConfig(
            seed=42,
            asset_class="sp500",
            encoder=encoder_type,
            min_horizon=self.config["min_horizon"],
            max_horizon=self.config["max_horizon"]
        )
        
        cfg = experiment_to_training_config(exp_config)
        
        # Override with test settings
        cfg.max_episodes = self.config["max_episodes"]
        cfg.val_episodes = self.config["val_episodes"]
        cfg.test_episodes = self.config["test_episodes"]
        cfg.val_interval = self.config["val_interval"]
        cfg.seq_len = self.config["seq_len"]
        cfg.batch_size = self.config["batch_size"]
        cfg.data_path = str(Path(self.temp_dir) / "test_sp500.parquet")
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        device = torch.device(cfg.device)
        
        # Prepare datasets
        datasets = create_split_datasets(cfg.data_path)
        train_dataset = datasets['train']
        cfg.num_assets = train_dataset.num_assets
        
        # Create environment
        seq_len_safe = min(self.config["seq_len"], len(train_dataset) - 1)
        window = train_dataset.get_window(0, seq_len_safe)
        mock_dataset = {
            'features': torch.tensor(window['features'], dtype=torch.float32),
            'raw_prices': torch.tensor(window['raw_prices'], dtype=torch.float32)
        }
        
        env = MetaEnv(
            dataset=mock_dataset,
            feature_columns=train_dataset.feature_cols,
            seq_len=seq_len_safe,
            min_horizon=cfg.min_horizon,
            max_horizon=cfg.max_horizon,
            eta=getattr(cfg, 'eta', 0.05),
            rf_rate=getattr(cfg, 'rf_rate', 0.02),
            transaction_cost_rate=getattr(cfg, 'transaction_cost_rate', 0.001)
        )
        
        # Get observation shape
        task = env.sample_task()
        env.set_task(task)
        obs_shape = env.reset().shape
        
        # Create models
        encoder = None
        if encoder_type == "vae":
            encoder = VAE(
                obs_dim=obs_shape,
                num_assets=cfg.num_assets,
                latent_dim=cfg.latent_dim,
                hidden_dim=cfg.hidden_dim
            ).to(device)
        elif encoder_type == "hmm":
            encoder = HMMEncoder(
                obs_dim=obs_shape,
                num_assets=cfg.num_assets,
                latent_dim=4,  # HMM uses 4 states
                hidden_dim=cfg.hidden_dim
            ).to(device)
        
        policy = PortfolioPolicy(
            obs_shape=obs_shape,
            latent_dim=cfg.latent_dim if encoder_type != "hmm" else 4,
            num_assets=cfg.num_assets,
            hidden_dim=cfg.hidden_dim
        ).to(device)
        
        # Create trainer
        trainer = PPOTrainer(env=env, policy=policy, vae=encoder, config=cfg)
        
        # Run minimal training
        episodes_trained = 0
        for episode in range(cfg.max_episodes):
            task = env.sample_task()
            env.set_task(task)
            
            result = trainer.train_episode()
            episodes_trained += 1
            
            # Validate that training produces reasonable outputs
            assert 'episode_reward' in result, "Missing episode_reward in training result"
            assert isinstance(result['episode_reward'], (int, float)), "episode_reward should be numeric"
            
            if episode % cfg.val_interval == 0 and episode > 0:
                # Quick validation check
                break
        
        assert episodes_trained > 0, "No episodes were trained"
        logger.info(f"Training loop with {encoder_type} encoder completed {episodes_trained} episodes")
    
    def test_evaluation_functionality(self):
        """Test evaluation and backtesting functionality"""
        logger.info("Testing evaluation functionality...")
        
        from evaluation_backtest import evaluate
        from config import ExperimentConfig, experiment_to_training_config
        from environments.dataset import create_split_datasets
        from environments.env import MetaEnv
        from models.policy import PortfolioPolicy
        from models.vae import VAE
        
        # Setup minimal config
        exp_config = ExperimentConfig(seed=42, asset_class="sp500", encoder="vae")
        cfg = experiment_to_training_config(exp_config)
        cfg.data_path = str(Path(self.temp_dir) / "test_sp500.parquet")
        cfg.seq_len = self.config["seq_len"]
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        device = torch.device(cfg.device)
        
        # Prepare test environment
        datasets = create_split_datasets(cfg.data_path)
        test_dataset = datasets['test']
        cfg.num_assets = test_dataset.num_assets
        
        seq_len_safe = min(cfg.seq_len, len(test_dataset) - 1)  # Ensure seq_len < dataset length
        window = test_dataset.get_window(0, seq_len_safe)
        mock_dataset = {
            'features': torch.tensor(window['features'], dtype=torch.float32),
            'raw_prices': torch.tensor(window['raw_prices'], dtype=torch.float32)
        }
        
        env = MetaEnv(
            dataset=mock_dataset,
            feature_columns=test_dataset.feature_cols,
            seq_len=seq_len_safe,  # Use safe sequence length
            min_horizon=max(1, getattr(cfg, 'min_horizon', 5)),  # Safe defaults
            max_horizon=max(2, getattr(cfg, 'max_horizon', 10))
        )
        
        # Create models (minimal)
        task = env.sample_task()
        env.set_task(task)
        obs_shape = env.reset().shape
        
        encoder = VAE(
            obs_dim=obs_shape,
            num_assets=cfg.num_assets,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim
        ).to(device)
        
        policy = PortfolioPolicy(
            obs_shape=obs_shape,
            latent_dim=cfg.latent_dim,
            num_assets=cfg.num_assets,
            hidden_dim=cfg.hidden_dim
        ).to(device)
        
        # Test evaluation
        eval_results = evaluate(env, policy, encoder, cfg, num_episodes=2)
        
        # Verify evaluation results
        required_metrics = ['avg_reward', 'std_reward', 'avg_return', 'num_episodes']
        for metric in required_metrics:
            assert metric in eval_results, f"Missing {metric} in evaluation results"
        
        assert eval_results['num_episodes'] == 2, "Wrong number of evaluation episodes"
        
        logger.info("Evaluation functionality tested successfully")
    
    def test_mlflow_integration(self):
        """Test MLflow setup and basic logging"""
        if not self.enable_mlflow:
            logger.info("MLflow testing skipped (disabled)")
            return
        
        logger.info("Testing MLflow integration...")
        
        try:
            from mlflow_logger import setup_mlflow, test_mlflow_connection
            
            # Test MLflow setup
            backend = setup_mlflow()
            assert backend in ['local', 'remote'], f"Invalid backend: {backend}"
            
            # Test connection
            connection_ok = test_mlflow_connection()
            assert connection_ok, "MLflow connection test failed"
            
            logger.info(f"MLflow integration tested successfully (backend: {backend})")
            
        except ImportError as e:
            logger.warning(f"MLflow modules not available: {e}")
            # This is acceptable in some test environments
    
    def run_all_tests(self) -> bool:
        """Run all smoke tests"""
        logger.info(f"Starting comprehensive smoke test (quick_mode={self.quick_mode})")
        logger.info(f"Temp directory: {self.temp_dir}")
        
        # Core functionality tests
        self.run_test("imports", self.test_imports)
        self.run_test("sp500_dataset_creation", self.test_sp500_dataset_creation)
        self.run_test("crypto_dataset_creation", self.test_crypto_dataset_creation)
        self.run_test("dataset_loading_and_splits", self.test_dataset_loading_and_splits)
        self.run_test("environment_creation", self.test_environment_creation)
        
        # Model tests
        self.run_test("model_creation_vae", self.test_model_creation_vae)
        self.run_test("model_creation_hmm", self.test_model_creation_hmm)
        self.run_test("model_creation_policy", self.test_model_creation_policy)
        
        # Training tests (most time-consuming)
        self.run_test("training_loop_vae", self.test_training_loop_vae)
        self.run_test("training_loop_none", self.test_training_loop_none)
        self.run_test("training_loop_hmm", self.test_training_loop_hmm)
        
        # Evaluation tests
        self.run_test("evaluation_functionality", self.test_evaluation_functionality)
        
        # Integration tests
        if self.enable_mlflow:
            self.run_test("mlflow_integration", self.test_mlflow_integration)
        
        # Print summary and return success status
        return self.results.print_summary()


def main():
    """Main entry point for smoke test"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Portfolio Optimization Pipeline Smoke Test")
    parser.add_argument("--quick", action="store_true", 
                       help="Run in quick mode (fewer episodes, smaller datasets)")
    parser.add_argument("--no-mlflow", action="store_true",
                       help="Skip MLflow integration tests")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging with full tracebacks")
    
    args = parser.parse_args()
    
    setup_test_logging(args.verbose)
    
    # Run smoke test
    with SmokeTest(
        quick_mode=args.quick,
        enable_mlflow=not args.no_mlflow,
        verbose=args.verbose
    ) as smoke_test:
        success = smoke_test.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()