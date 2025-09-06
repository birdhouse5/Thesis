# smoke_test_simple.py - Quick test to verify training works end-to-end

import torch
import logging
import numpy as np
from pathlib import Path
import time

from main import ValidationConfig, ExperimentRunner

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def smoke_test():
    """Quick smoke test with minimal settings"""
    print("🧪 Running smoke test...")
    
    # Super minimal config for fast testing
    config = ValidationConfig(
        seed=42,
        exp_name="smoke_test",
        
        # Minimal training
        max_episodes=6,           # Just 6 episodes total
        episodes_per_task=1,      # 1 episode per task
        val_interval=3,           # Validate every 3 episodes
        val_episodes=2,           # Only 2 validation episodes
        
        # Small parameters
        batch_size=64,            # Tiny batch
        vae_batch_size=8,         # Tiny VAE batch
        
        # Fast settings
        num_envs=1,               # Single environment
        debug_mode=False,         # No debug spam
        
        # Keep other settings normal
        min_horizon=50,
        max_horizon=50,           # Fixed length
    )
    
    try:
        print(f"Config: {config.max_episodes} episodes, {config.episodes_per_task} per task")
        print(f"Validation every {config.val_interval} episodes")
        
        # Initialize runner
        runner = ExperimentRunner("smoke_test_results")
        
        # Setup data (this takes a moment)
        print("Setting up data environment...")
        start_time = time.time()
        runner.setup_data_environment(config)
        setup_time = time.time() - start_time
        print(f"Data setup took {setup_time:.1f}s")
        
        # Run single seed
        print("Running training...")
        start_time = time.time()
        result = runner.run_single_seed(config)
        training_time = time.time() - start_time
        
        print(f"Training took {training_time:.1f}s")
        print(f"Episodes trained: {result.episodes_trained}")
        print(f"Best val sharpe: {result.best_val_sharpe:.4f}")
        print(f"Early stopped: {result.early_stopped}")
        
        # Verify the result makes sense
        assert result.episodes_trained > 0, "No episodes were trained"
        assert result.episodes_trained <= config.max_episodes, "Too many episodes trained"
        assert result.best_val_sharpe != float('-inf'), "No validation occurred"
        
        print("✅ Smoke test PASSED - training pipeline works!")
        return True
        
    except Exception as e:
        print(f"❌ Smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def quick_buffer_test():
    """Test just the ExperienceBuffer to verify the fix"""
    print("🧪 Testing ExperienceBuffer...")
    
    try:
        from algorithms.trainer import ExperienceBuffer
        
        # Create buffer
        buffer = ExperienceBuffer(min_batch_size=2)
        
        # Add dummy trajectory
        dummy_traj = {
            "observations": torch.randn(10, 30, 25),
            "actions": torch.randn(10, 30),
            "rewards": torch.randn(10),
            "values": torch.randn(10),
            "log_probs": torch.randn(10),
            "latents": torch.randn(10, 512),
            "dones": [False] * 9 + [True]
        }
        
        buffer.add_trajectory(dummy_traj)
        print(f"Buffer has {buffer.total_steps} steps")
        
        # Test if ready
        is_ready = buffer.is_ready()
        print(f"Buffer ready: {is_ready}")
        
        # Test get_all
        all_trajs = buffer.get_all()
        print(f"Retrieved {len(all_trajs)} trajectories")
        
        # Test clear (this was the failing method)
        buffer.clear()
        print(f"After clear: {buffer.total_steps} steps")
        
        print("✅ ExperienceBuffer test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ ExperienceBuffer test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("SMOKE TESTING PIPELINE")
    print("=" * 50)
    
    # Test 1: Buffer functionality
    buffer_ok = quick_buffer_test()
    print()
    
    # Test 2: Full pipeline if buffer works
    if buffer_ok:
        pipeline_ok = smoke_test()
    else:
        print("Skipping pipeline test due to buffer failure")
        pipeline_ok = False
    
    print()
    print("=" * 50)
    print("SMOKE TEST SUMMARY")
    print("=" * 50)
    print(f"ExperienceBuffer: {'✅ PASS' if buffer_ok else '❌ FAIL'}")
    print(f"Training Pipeline: {'✅ PASS' if pipeline_ok else '❌ FAIL'}")
    
    if buffer_ok and pipeline_ok:
        print("\n🎉 All tests passed! Ready for full experiment.")
        print("Run: python main.py")
    else:
        print("\n🔧 Fix required before running full experiment.")