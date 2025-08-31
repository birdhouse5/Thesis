#!/usr/bin/env python3
"""
Smoke test for final study implementation - verifies proper seed isolation and config loading.
This test runs minimal training to verify the randomization works correctly without burning compute.

Usage:
    python smoke_test.py --quick     # Fast test with minimal episodes
    python smoke_test.py --detailed  # More thorough test
"""

import torch
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, List, Any, Tuple

# Import the modules we're testing
from run_logger import seed_everything

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_seed_isolation():
    """Test that different seeds produce different results"""
    logger.info("Testing seed isolation...")
    
    results = {}
    seeds = [0, 1, 2, 3, 4]
    
    for seed in seeds:
        seed_everything(seed)
        
        # Test basic randomness
        torch_rand = torch.randn(5).tolist()
        numpy_rand = np.random.randn(5).tolist()
        
        results[seed] = {
            'torch_rand': torch_rand,
            'numpy_rand': numpy_rand
        }
        
        logger.info(f"Seed {seed}: torch={torch_rand[0]:.6f}, numpy={numpy_rand[0]:.6f}")
    
    # Verify all results are different
    torch_values = [results[seed]['torch_rand'][0] for seed in seeds]
    numpy_values = [results[seed]['numpy_rand'][0] for seed in seeds]
    
    torch_unique = len(set(f"{v:.6f}" for v in torch_values))
    numpy_unique = len(set(f"{v:.6f}" for v in numpy_values))
    
    if torch_unique == len(seeds) and numpy_unique == len(seeds):
        logger.info("✅ PASS: Seed isolation working - all seeds produce different results")
        return True
    else:
        logger.error(f"❌ FAIL: Seed isolation broken - torch_unique={torch_unique}, numpy_unique={numpy_unique}")
        return False


def test_config_loading():
    """Test that different trial configs are actually different"""
    logger.info("Testing configuration loading...")
    
    try:
        # Import the config loading function
        from main import load_top5_configs
        
        configs = load_top5_configs("experiment_configs")
        
        if len(configs) != 5:
            logger.warning(f"Expected 5 configs, got {len(configs)}")
        
        # Check if configs are actually different
        unique_params = set()
        param_variations = {}
        
        for config in configs:
            trial_id = config.get('trial_id', 'unknown')
            
            # Key parameters that should vary between trials
            key_params = (
                config.get('vae_lr', 0),
                config.get('policy_lr', 0),
                config.get('latent_dim', 0),
                config.get('hidden_dim', 0),
                config.get('batch_size', 0)
            )
            
            unique_params.add(key_params)
            
            # Track parameter variations
            for param in ['vae_lr', 'policy_lr', 'latent_dim', 'hidden_dim', 'batch_size']:
                if param not in param_variations:
                    param_variations[param] = set()
                param_variations[param].add(config.get(param))
            
            logger.info(f"Trial {trial_id}: vae_lr={config.get('vae_lr'):.6f}, "
                       f"latent_dim={config.get('latent_dim')}, batch_size={config.get('batch_size')}")
        
        # Analysis
        if len(unique_params) == len(configs):
            logger.info("✅ PASS: All configurations are unique")
        else:
            logger.warning(f"⚠️  Only {len(unique_params)}/{len(configs)} unique configurations")
        
        # Check parameter diversity
        for param, values in param_variations.items():
            logger.info(f"Parameter {param}: {len(values)} unique values = {sorted(values)}")
        
        return len(unique_params) > 1, configs
        
    except Exception as e:
        logger.error(f"❌ FAIL: Config loading failed: {e}")
        return False, []


def test_model_initialization_variation(configs: List[Dict], quick_mode: bool = True):
    """Test that models initialized with different seeds have different weights"""
    logger.info("Testing model initialization variation...")
    
    try:
        # Import required modules
        from main import StudyConfig, initialize_models
        from environments.dataset import create_split_datasets
        
        # Use first config as template
        base_config_dict = configs[0] if configs else {
            'trial_id': 69,
            'vae_lr': 0.001,
            'policy_lr': 0.002,
            'latent_dim': 512,
            'hidden_dim': 1024
        }
        
        # Test with different seeds
        seeds = [0, 1, 2]
        model_weights = {}
        
        for seed in seeds:
            # Create config for this seed
            config = StudyConfig(
                trial_id=base_config_dict['trial_id'],
                seed=seed,
                exp_name=f"test_seed_{seed}"
            )
            
            # Update with parameters
            for key, value in base_config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            # Set seed
            seed_everything(seed)
            
            # Initialize models
            obs_shape = (30, 25)  # Standard shape from logs
            vae, policy = initialize_models(config, obs_shape)
            
            # Extract a few key weights
            vae_weight = vae.encoder.obs_encoder[0].weight[0, 0].item()
            policy_weight = policy.obs_encoder[0].weight[0, 0].item()
            
            model_weights[seed] = {
                'vae_weight': vae_weight,
                'policy_weight': policy_weight
            }
            
            logger.info(f"Seed {seed}: VAE={vae_weight:.6f}, Policy={policy_weight:.6f}")
        
        # Check uniqueness
        vae_weights = [model_weights[seed]['vae_weight'] for seed in seeds]
        policy_weights = [model_weights[seed]['policy_weight'] for seed in seeds]
        
        vae_unique = len(set(f"{w:.6f}" for w in vae_weights))
        policy_unique = len(set(f"{w:.6f}" for w in policy_weights))
        
        if vae_unique == len(seeds) and policy_unique == len(seeds):
            logger.info("✅ PASS: Model initialization produces different weights for different seeds")
            return True
        else:
            logger.error(f"❌ FAIL: Model weights not varying - vae_unique={vae_unique}, policy_unique={policy_unique}")
            return False
            
    except Exception as e:
        logger.error(f"❌ FAIL: Model initialization test failed: {e}")
        return False


def test_training_variation(configs: List[Dict], quick_mode: bool = True):
    """Test that training with different seeds produces different results"""
    logger.info("Testing training variation...")
    
    if quick_mode:
        logger.info("Quick mode: Testing with minimal training episodes")
    
    try:
        # Import training function
        from main import StudyConfig, train_single_run, prepare_datasets
        
        # Use first config as template
        base_config_dict = configs[0] if configs else {
            'trial_id': 69,
            'vae_lr': 0.001,
            'policy_lr': 0.002,
            'latent_dim': 512,
            'hidden_dim': 1024
        }
        
        # Prepare datasets once
        sample_config = StudyConfig(trial_id=base_config_dict['trial_id'], seed=0, exp_name="smoke_test")
        for key, value in base_config_dict.items():
            if hasattr(sample_config, key):
                setattr(sample_config, key, value)
        
        # Override for quick testing
        if quick_mode:
            sample_config.max_episodes = 20  # Minimal training
            sample_config.val_interval = 10  # Check very frequently
            sample_config.val_episodes = 5   # Minimal validation
        
        split_tensors, _ = prepare_datasets(sample_config)
        
        # Test with different seeds
        seeds = [0, 1] if quick_mode else [0, 1, 2]
        training_results = {}
        
        for seed in seeds:
            logger.info(f"Running minimal training with seed {seed}...")
            
            # Create config for this seed
            run_config = StudyConfig(
                trial_id=base_config_dict['trial_id'],
                seed=seed,
                exp_name=f"smoke_test_seed_{seed}"
            )
            
            # Update with parameters
            for key, value in base_config_dict.items():
                if hasattr(run_config, key):
                    setattr(run_config, key, value)
            
            # Apply quick mode settings
            if quick_mode:
                run_config.max_episodes = 20
                run_config.val_interval = 10
                run_config.val_episodes = 5
                run_config.early_stopping_patience = 2  # Stop quickly
            
            # Run training
            result = train_single_run(run_config, split_tensors)
            
            training_results[seed] = {
                'best_val_sharpe': result['best_val_sharpe'],
                'episodes_trained': result['episodes_trained'],
                'early_stopped': result['early_stopped']
            }
            
            logger.info(f"Seed {seed}: val_sharpe={result['best_val_sharpe']:.4f}, "
                       f"episodes={result['episodes_trained']}")
        
        # Analyze results
        sharpe_values = [training_results[seed]['best_val_sharpe'] for seed in seeds]
        episodes_values = [training_results[seed]['episodes_trained'] for seed in seeds]
        
        # Check for variation (allowing for some tolerance due to small episode count)
        sharpe_range = max(sharpe_values) - min(sharpe_values)
        episodes_vary = len(set(episodes_values)) > 1
        
        if sharpe_range > 0.01 or episodes_vary:  # Some variation expected
            logger.info("✅ PASS: Training produces different results for different seeds")
            logger.info(f"   Sharpe range: {sharpe_range:.4f}, Episode variation: {episodes_vary}")
            return True
        else:
            logger.error(f"❌ FAIL: Training results too similar - sharpe_range={sharpe_range:.4f}")
            logger.error("   This suggests seeds aren't affecting training randomness")
            return False
            
    except Exception as e:
        logger.error(f"❌ FAIL: Training variation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run comprehensive smoke test"""
    parser = argparse.ArgumentParser(description="Smoke test for final study implementation")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (minimal episodes)")
    parser.add_argument("--detailed", action="store_true", help="Detailed test mode")
    
    args = parser.parse_args()
    quick_mode = args.quick or not args.detailed
    
    logger.info("=" * 60)
    logger.info("FINAL STUDY SMOKE TEST")
    logger.info("=" * 60)
    logger.info(f"Mode: {'Quick' if quick_mode else 'Detailed'}")
    
    # Test results
    test_results = {
        'seed_isolation': False,
        'config_loading': False,
        'model_initialization': False,
        'training_variation': False
    }
    
    configs = []
    
    # Test 1: Seed Isolation
    logger.info("\n" + "=" * 40)
    logger.info("TEST 1: Seed Isolation")
    logger.info("=" * 40)
    test_results['seed_isolation'] = test_seed_isolation()
    
    # Test 2: Config Loading
    logger.info("\n" + "=" * 40)
    logger.info("TEST 2: Configuration Loading")
    logger.info("=" * 40)
    config_pass, configs = test_config_loading()
    test_results['config_loading'] = config_pass
    
    # Test 3: Model Initialization Variation
    logger.info("\n" + "=" * 40)
    logger.info("TEST 3: Model Initialization Variation")
    logger.info("=" * 40)
    test_results['model_initialization'] = test_model_initialization_variation(configs, quick_mode)
    
    # Test 4: Training Variation (most expensive)
    logger.info("\n" + "=" * 40)
    logger.info("TEST 4: Training Variation")
    logger.info("=" * 40)
    if quick_mode:
        logger.info("Running minimal training test (20 episodes)...")
    test_results['training_variation'] = test_training_variation(configs, quick_mode)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SMOKE TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name.replace('_', ' ').title()}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Implementation appears correct")
        logger.info("   You can proceed with the full study")
    else:
        logger.error("🚨 SOME TESTS FAILED - Fix issues before running full study")
        logger.error("   The main study will likely produce invalid results")
    
    # Recommendations
    logger.info("\n" + "=" * 40)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 40)
    
    if not test_results['seed_isolation']:
        logger.error("- Fix seed_everything() function - seeds aren't isolating properly")
    
    if not test_results['config_loading']:
        logger.error("- Check experiment_configs/ directory - may be missing or configs too similar")
    
    if not test_results['model_initialization']:
        logger.error("- Model initialization not varying with seeds - check seed application timing")
    
    if not test_results['training_variation']:
        logger.error("- Training not producing different results - check training loop randomness")
        
    if passed == total:
        logger.info("- Implementation looks solid!")
        logger.info("- Consider running full study with --run-final-study")
        logger.info("- Expected runtime: ~8-12 hours for 5 configs × 5 seeds")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)