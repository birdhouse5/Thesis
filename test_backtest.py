"""
Test script to verify backtest functionality works correctly.
Tests device handling and basic execution.
"""

import torch
import logging
from pathlib import Path

from config import ExperimentConfig, experiment_to_training_config
from main import prepare_environments, create_models
from evaluation_backtest import run_sequential_backtest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_backtest_device_handling():
    """Test that backtest handles device placement correctly."""
    
    logger.info("="*80)
    logger.info("TEST: Backtest Device Handling")
    logger.info("="*80)
    
    # Create minimal config
    exp = ExperimentConfig(
        seed=0,
        asset_class="sp500",
        encoder="vae",
        n_assets=5  # Use only 5 assets for speed
    )
    cfg = experiment_to_training_config(exp)
    
    # Override for faster testing
    cfg.max_episodes = 10
    cfg.seq_len = 50  # Shorter sequences
    
    logger.info(f"\nConfig:")
    logger.info(f"  Asset class: {cfg.asset_class}")
    logger.info(f"  Encoder: {cfg.encoder}")
    logger.info(f"  Device: {cfg.device}")
    logger.info(f"  N assets: {cfg.n_assets_limit}")
    
    try:
        # Prepare environments
        logger.info("\n1. Loading dataset and creating environments...")
        envs, split_tensors, datasets = prepare_environments(cfg)
        logger.info(f"   ✓ Datasets loaded")
        logger.info(f"   ✓ Train: {len(datasets['train'])} days")
        logger.info(f"   ✓ Val: {len(datasets['val'])} days")
        logger.info(f"   ✓ Test: {len(datasets['test'])} days")
        
        # Create models
        logger.info("\n2. Creating models...")
        task = envs['train'].sample_task()
        envs['train'].set_task(task)
        obs_shape = envs['train'].reset().shape
        
        encoder, policy = create_models(cfg, obs_shape)
        logger.info(f"   ✓ Encoder created: {type(encoder).__name__ if encoder else 'None'}")
        logger.info(f"   ✓ Policy created on device: {next(policy.parameters()).device}")
        
        # Verify device placement
        policy_device = next(policy.parameters()).device
        env_device = envs['test'].device
        logger.info(f"\n3. Checking device placement...")
        logger.info(f"   Policy device: {policy_device}")
        logger.info(f"   Environment device: {env_device}")
        
        # Run backtest (this is where the error would occur)
        logger.info("\n4. Running sequential backtest on test split...")
        logger.info(f"   Test split size: {len(datasets['test'])} days")
        
        backtest_results = run_sequential_backtest(
            datasets, 
            policy, 
            encoder, 
            cfg, 
            split='test'
        )
        
        logger.info("\n5. ✅ BACKTEST COMPLETED SUCCESSFULLY!")
        
        # Print summary results
        logger.info("\n" + "="*80)
        logger.info("BACKTEST RESULTS SUMMARY")
        logger.info("="*80)
        logger.info(f"Total Return: {backtest_results['total_return']:.2%}")
        logger.info(f"Annual Return: {backtest_results['annual_return']:.2%}")
        logger.info(f"Annual Volatility: {backtest_results['annual_volatility']:.2%}")
        logger.info(f"Sharpe Ratio: {backtest_results['backtest_sharpe']:.3f}")
        logger.info(f"Max Drawdown: {backtest_results['max_drawdown']:.2%}")
        logger.info(f"Win Rate: {backtest_results['win_rate']:.2%}")
        logger.info(f"Number of Trades: {backtest_results['num_trades']}")
        logger.info(f"Final Capital: ${backtest_results['final_capital']:,.2f}")
        logger.info("="*80)
        
        # Verify no NaN/Inf values
        logger.info("\n6. Verifying result validity...")
        for key, value in backtest_results.items():
            if isinstance(value, (int, float)):
                if value != value:  # NaN check
                    logger.error(f"   ✗ NaN detected in {key}")
                    return False
                if abs(value) == float('inf'):
                    logger.error(f"   ✗ Inf detected in {key}")
                    return False
        
        logger.info("   ✓ All results are valid (no NaN/Inf)")
        
        # Check CSV output
        logger.info("\n7. Checking CSV output...")
        csv_path = Path(cfg.exp_name) / cfg.encoder / cfg.asset_class / "experiment_logs" / f"{cfg.exp_name}_backtest.csv"
        if csv_path.exists():
            logger.info(f"   ✓ CSV file created: {csv_path}")
            
            # Check file size
            file_size = csv_path.stat().st_size / 1024  # KB
            logger.info(f"   ✓ CSV file size: {file_size:.1f} KB")
            
            # Verify it has data
            import pandas as pd
            df = pd.read_csv(csv_path)
            logger.info(f"   ✓ CSV has {len(df)} rows")
            logger.info(f"   ✓ CSV columns: {list(df.columns)[:5]}...")
        else:
            logger.warning(f"   ⚠ CSV file not found at {csv_path}")
        
        logger.info("\n" + "="*80)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("="*80)
        
        return True
        
    except RuntimeError as e:
        if "Expected all tensors to be on the same device" in str(e):
            logger.error("\n" + "="*80)
            logger.error("❌ DEVICE MISMATCH ERROR DETECTED")
            logger.error("="*80)
            logger.error(f"Error: {str(e)}")
            logger.error("\nThis indicates the fix was not applied correctly.")
            logger.error("Please ensure evaluation_backtest.py line ~277 has:")
            logger.error("    action = action.to(env.device)")
            return False
        else:
            logger.error(f"\n❌ Unexpected error: {str(e)}")
            raise
    
    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_backtest_with_different_devices():
    """Test backtest works with both CPU and GPU configurations."""
    
    logger.info("\n" + "="*80)
    logger.info("TEST: Multiple Device Configurations")
    logger.info("="*80)
    
    devices_to_test = ['cpu']
    if torch.cuda.is_available():
        devices_to_test.append('cuda')
        logger.info("✓ CUDA available, testing both CPU and GPU")
    else:
        logger.info("⚠ CUDA not available, testing CPU only")
    
    results = {}
    
    for device_name in devices_to_test:
        logger.info(f"\n--- Testing with device: {device_name} ---")
        
        try:
            exp = ExperimentConfig(
                seed=0,
                asset_class="sp500",
                encoder="none",  # Faster without VAE
                n_assets=5
            )
            cfg = experiment_to_training_config(exp)
            cfg.device = device_name
            cfg.seq_len = 50
            
            # Quick environment setup
            envs, split_tensors, datasets = prepare_environments(cfg)
            task = envs['train'].sample_task()
            envs['train'].set_task(task)
            obs_shape = envs['train'].reset().shape
            encoder, policy = create_models(cfg, obs_shape)
            
            # Run minimal backtest (just first 100 steps)
            logger.info(f"   Running backtest on {device_name}...")
            
            # Monkey-patch to run only 100 steps for speed
            original_len = len(datasets['test'])
            
            backtest_results = run_sequential_backtest(
                datasets, policy, encoder, cfg, split='test'
            )
            
            results[device_name] = {
                'success': True,
                'return': backtest_results['total_return'],
                'sharpe': backtest_results['backtest_sharpe']
            }
            
            logger.info(f"   ✓ {device_name.upper()} backtest successful")
            logger.info(f"     Return: {backtest_results['total_return']:.2%}")
            logger.info(f"     Sharpe: {backtest_results['backtest_sharpe']:.3f}")
            
        except Exception as e:
            logger.error(f"   ✗ {device_name.upper()} backtest failed: {str(e)}")
            results[device_name] = {'success': False, 'error': str(e)}
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("DEVICE TEST SUMMARY")
    logger.info("="*80)
    
    all_passed = all(r['success'] for r in results.values())
    
    for device_name, result in results.items():
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        logger.info(f"{device_name.upper()}: {status}")
        if not result['success']:
            logger.info(f"  Error: {result['error']}")
    
    logger.info("="*80)
    
    if all_passed:
        logger.info("✅ ALL DEVICE CONFIGURATIONS PASSED")
    else:
        logger.error("❌ SOME DEVICE CONFIGURATIONS FAILED")
    
    return all_passed


if __name__ == "__main__":
    logger.info("\n" + "="*80)
    logger.info("BACKTEST VERIFICATION TEST SUITE")
    logger.info("="*80)
    
    # Run tests
    test1_passed = test_backtest_device_handling()
    
    # Only run device tests if first test passed
    if test1_passed:
        test2_passed = test_backtest_with_different_devices()
        all_passed = test1_passed and test2_passed
    else:
        logger.error("\nSkipping device tests due to basic test failure")
        all_passed = False
    
    # Final summary
    logger.info("\n" + "="*80)
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED - BACKTEST IS WORKING CORRECTLY")
    else:
        logger.error("❌ SOME TESTS FAILED - PLEASE REVIEW ERRORS ABOVE")
    logger.info("="*80)
    
    exit(0 if all_passed else 1)