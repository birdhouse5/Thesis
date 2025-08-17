#!/usr/bin/env python3
"""
Smoke test for Phase 3 - runs 1 trial with minimal settings to verify everything works.
Usage: python smoke_test_phase3.py
"""

import torch
torch.set_num_threads(1)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    
import optuna
import logging
import numpy as np
from pathlib import Path
import json
import time

# Import your modules
from environments.dataset import create_split_datasets
from environments.env import MetaEnv
from models.vae import VAE
from models.policy import PortfolioPolicy
from algorithms.trainer import PPOTrainer
from run_logger import RunLogger, seed_everything

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmokeTestConfig:
    """Minimal config for smoke testing"""
    def __init__(self, trial=None):
        # Fixed Phase 2 parameters (minimal for speed)
        self.latent_dim = 64  # Smaller for speed
        self.hidden_dim = 128  # Smaller for speed
        self.vae_lr = 0.001
        self.policy_lr = 0.002
        self.vae_beta = 0.01
        self.vae_update_freq = 5
        self.seq_len = 30  # Much smaller for speed
        self.episodes_per_task = 2
        self.batch_size = 256  # Smaller for speed
        self.vae_batch_size = 128
        self.ppo_epochs = 2
        self.entropy_coef = 0.01
        
        # MINIMAL early stopping test
        if trial:
            self.max_episodes = trial.suggest_categorical('max_episodes', [100, 200])
            self.early_stopping_patience = trial.suggest_categorical('early_stopping_patience', [2, 3])
            self.early_stopping_min_delta = trial.suggest_categorical('early_stopping_min_delta', [0.01, 0.05])
            self.val_interval = trial.suggest_categorical('val_interval', [50, 100])
        else:
            # Default values for single test
            self.max_episodes = 200
            self.early_stopping_patience = 3
            self.early_stopping_min_delta = 0.01
            self.val_interval = 50
        
        # Environment settings
        self.data_path = "environments/data/sp500_rl_ready_cleaned.parquet"
        self.train_end = '2015-12-31'
        self.val_end = '2020-12-31'
        self.num_assets = 30
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Derived settings
        self.max_horizon = min(self.seq_len - 5, int(self.seq_len * 0.8))
        self.min_horizon = max(self.max_horizon - 10, self.max_horizon // 2)
        self.min_episodes_before_stopping = 50  # Very low for testing
        
        # Minimal validation
        self.val_episodes = 5  # Very small for speed
        self.log_interval = 20
        self.save_interval = 1000
        
        # Fixed parameters
        self.ppo_clip_ratio = 0.2
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.gae_lambda = 0.95
        self.discount_factor = 0.99
        
        self.seed = 42
        self.exp_name = "smoke_test"

def smoke_test_single():
    """Test single run without Optuna"""
    print("🧪 SINGLE RUN SMOKE TEST")
    print("="*40)
    
    config = SmokeTestConfig()
    seed_everything(config.seed)
    
    # Create minimal run directory
    run_dir = Path("smoke_test_results")
    run_dir.mkdir(exist_ok=True)
    run = RunLogger(run_dir, config.__dict__, name=config.exp_name)
    
    try:
        print("1. Loading datasets...")
        if not Path(config.data_path).exists():
            print("❌ Dataset not found. Please run data preparation first.")
            return False
        
        datasets = create_split_datasets(
            data_path=config.data_path,
            train_end=config.train_end,
            val_end=config.val_end
        )
        print("✅ Datasets loaded")
        
        print("2. Creating environments...")
        # Minimal environment setup
        train_dataset = datasets['train']
        val_dataset = datasets['val']
        
        # Create larger windows for environment (need enough data for task sampling)
        window_size = max(100, config.seq_len * 3)  # Ensure enough data
        train_window = train_dataset.get_window(0, min(window_size, len(train_dataset)))
        val_window = val_dataset.get_window(0, min(window_size, len(val_dataset)))
        
        train_tensor = {
            'features': torch.tensor(train_window['features'], dtype=torch.float32),
            'raw_prices': torch.tensor(train_window['raw_prices'], dtype=torch.float32)
        }
        val_tensor = {
            'features': torch.tensor(val_window['features'], dtype=torch.float32),
            'raw_prices': torch.tensor(val_window['raw_prices'], dtype=torch.float32)
        }
        
        train_env = MetaEnv(train_tensor, train_dataset.feature_cols, config.seq_len, config.min_horizon, config.max_horizon)
        val_env = MetaEnv(val_tensor, val_dataset.feature_cols, config.seq_len, config.min_horizon, config.max_horizon)
        print("✅ Environments created")
        
        print("3. Initializing models...")
        task = train_env.sample_task()
        train_env.set_task(task)
        initial_obs = train_env.reset()
        obs_shape = initial_obs.shape
        
        device = torch.device(config.device)
        vae = VAE(obs_shape, config.num_assets, latent_dim=config.latent_dim, hidden_dim=config.hidden_dim).to(device)
        policy = PortfolioPolicy(obs_shape, config.latent_dim, config.num_assets, config.hidden_dim).to(device)
        print("✅ Models initialized")
        
        print("4. Testing trainer with early stopping...")
        trainer = PPOTrainer(env=train_env, policy=policy, vae=vae, config=config)
        
        # Test early stopping methods exist
        assert hasattr(trainer, 'add_validation_score'), "Missing add_validation_score method"
        assert hasattr(trainer, 'should_stop_early'), "Missing should_stop_early method"
        assert hasattr(trainer, 'get_early_stopping_state'), "Missing get_early_stopping_state method"
        print("✅ Early stopping methods found")
        
        print("5. Testing training loop...")
        episodes_trained = 0
        for i in range(3):  # Just 3 iterations
            episode_result = trainer.train_episode()
            episodes_trained += 1
            
            run.log_train_episode(episodes_trained, reward=episode_result.get('episode_reward', 0))
            
            if episodes_trained % 2 == 0:  # Test validation
                # Quick validation
                val_score = np.random.uniform(0, 1)  # Mock score for speed
                should_stop = trainer.add_validation_score(val_score)
                
                run.log_val(episodes_trained, sharpe=val_score, reward=val_score, cum_wealth=0.0)
                
                print(f"  Episode {episodes_trained}: val_score={val_score:.4f}, should_stop={should_stop}")
        
        print("✅ Training loop works")
        
        print("6. Testing early stopping state...")
        state = trainer.get_early_stopping_state()
        assert 'validation_scores' in state
        assert 'best_val_score' in state
        print("✅ Early stopping state works")
        
        run.close()
        print("\n🎉 SINGLE RUN SMOKE TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ SINGLE RUN SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        run.close()
        return False

def smoke_test_optuna():
    """Test Optuna integration"""
    print("\n🧪 OPTUNA SMOKE TEST")
    print("="*40)
    
    def objective(trial):
        config = SmokeTestConfig(trial)
        seed_everything(config.seed + trial.number)
        
        # Super minimal validation
        episodes_trained = 0
        mock_val_scores = []
        
        # Simulate training with early stopping
        while episodes_trained < config.max_episodes:
            episodes_trained += config.val_interval
            
            # Mock validation score (declining to test early stopping)
            mock_score = max(0.1, 1.0 - episodes_trained * 0.01)
            mock_val_scores.append(mock_score)
            
            # Simple early stopping simulation
            if len(mock_val_scores) >= config.early_stopping_patience:
                recent_scores = mock_val_scores[-config.early_stopping_patience:]
                if all(s < mock_val_scores[0] + config.early_stopping_min_delta for s in recent_scores[1:]):
                    print(f"  Mock early stopping at episode {episodes_trained}")
                    break
            
            # Report to Optuna
            trial.report(mock_score, episodes_trained)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return max(mock_val_scores) if mock_val_scores else 0.1
    
    try:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=2, timeout=60)  # 2 trials, 1 minute max
        
        print(f"✅ Optuna completed {len(study.trials)} trials")
        if study.best_trial:
            print(f"✅ Best trial value: {study.best_trial.value:.4f}")
            print(f"✅ Best params: {study.best_trial.params}")
        
        print("\n🎉 OPTUNA SMOKE TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ OPTUNA SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run both smoke tests"""
    print("🧪 PHASE 3 SMOKE TESTS")
    print("="*50)
    print("Testing minimal version before 8-hour run...")
    
    # Test 1: Single run
    single_passed = smoke_test_single()
    
    # Test 2: Optuna integration
    optuna_passed = smoke_test_optuna()
    
    # Summary
    print("\n" + "="*50)
    print("SMOKE TEST SUMMARY")
    print("="*50)
    print(f"Single run test: {'✅ PASS' if single_passed else '❌ FAIL'}")
    print(f"Optuna test: {'✅ PASS' if optuna_passed else '❌ FAIL'}")
    
    if single_passed and optuna_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Safe to run full Phase 3 optimization")
        print("Run: python optuna_phase3.py")
    else:
        print("\n❌ TESTS FAILED!")
        print("Fix issues before running full optimization")
    
    print(f"\nTime taken: ~{time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    start_time = time.time()
    main()