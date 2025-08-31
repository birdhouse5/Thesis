#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Make numpy print stable-ish
np.set_printoptions(precision=6, suppress=True)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def banner(title: str):
    logger.info("\n" + "=" * 60)
    logger.info(title)
    logger.info("=" * 60)


def sec(title: str):
    logger.info("\n" + "=" * 40)
    logger.info(title)
    logger.info("=" * 40)


def fingerprint_model(model: torch.nn.Module, max_params: int = 5) -> Dict[str, float]:
    """
    Produce a seed-sensitive but stable fingerprint from the first few parameter tensors
    without assuming any specific submodule structure.
    """
    sums, means, stds = [], [], []
    count = 0
    for p in model.parameters():
        if p is None:
            continue
        t = p.detach().float().flatten()
        if t.numel() == 0:
            continue
        # sample up to 10k elements for speed
        if t.numel() > 10000:
            idx = torch.linspace(0, t.numel() - 1, steps=10000).long()
            t = t.index_select(0, idx)
        sums.append(float(t.sum().item()))
        means.append(float(t.mean().item()))
        stds.append(float(t.std().item()))
        count += 1
        if count >= max_params:
            break
    return {
        "count": count,
        "sum": float(np.sum(sums)) if sums else 0.0,
        "mean": float(np.mean(means)) if means else 0.0,
        "std": float(np.mean(stds)) if stds else 0.0,
    }


def unique_values_summary(configs: List[Dict[str, Any]], keys: List[str]) -> Dict[str, List[Any]]:
    out = {}
    for k in keys:
        vals = []
        for c in configs:
            if k in c:
                vals.append(c[k])
        out[k] = sorted(list(set(vals)), key=lambda x: json.dumps(x, sort_keys=True))
    return out


def patch_torch_as_tensor():
    """
    Monkey-patch torch.as_tensor to tolerate an unexpected 'non_blocking' kwarg.
    This is safe for the smoke test and avoids modifying your training code.
    """
    orig = torch.as_tensor

    def _as_tensor_safe(data, *args, **kwargs):
        if "non_blocking" in kwargs:
            kwargs.pop("non_blocking", None)
        return orig(data, *args, **kwargs)

    torch.as_tensor = _as_tensor_safe  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_seed_isolation() -> bool:
    sec("TEST 1: Seed Isolation")
    logger.info("Testing seed isolation...")

    results = []
    for s in range(5):
        torch.manual_seed(s)
        np.random.seed(s)
        t = float(torch.randn(1).item())
        n = float(np.random.randn(1)[0])
        logger.info(f"Seed {s}: torch={t:.6f}, numpy={n:.6f}")
        results.append((t, n))

    # Simple: just ensure not all identical
    all_same = all(abs(results[i][0] - results[0][0]) < 1e-8 and abs(results[i][1] - results[0][1]) < 1e-8 for i in range(1, 5))
    if not all_same:
        logger.info("✅ PASS: Seed isolation working - all seeds produce different results")
        return True
    else:
        logger.error("❌ FAIL: Seed isolation not working")
        return False


def test_config_loading(configs_dir: str) -> bool:
    sec("TEST 2: Configuration Loading")
    logger.info("Testing configuration loading...")

    # Check expected filenames exist (best practice)
    expected = [69, 9, 26, 54, 5]
    missing = []
    for tid in expected:
        if not (Path(configs_dir) / f"config_{tid}.json").exists():
            missing.append(tid)
    if missing:
        logger.warning(f"⚠️  Missing exact config files for trial(s): {missing} "
                       f"(expected files like config_{{trial}}.json).")

    # Import loader from main
    try:
        import main as mainmod
    except Exception as e:
        logger.error(f"❌ FAIL: Could not import main.py: {e}")
        return False

    try:
        cfgs = mainmod.load_top5_configs(configs_dir)
    except Exception as e:
        logger.error(f"❌ FAIL: load_top5_configs raised: {e}")
        return False

    # Summarize uniqueness
    keys_to_check = [
        "vae_lr", "policy_lr", "latent_dim", "hidden_dim",
        "batch_size", "vae_beta", "vae_update_freq", "seq_len",
        "episodes_per_task", "ppo_epochs", "entropy_coef"
    ]
    uniq = unique_values_summary(cfgs, keys_to_check)
    unique_counts = {k: len(v) for k, v in uniq.items()}

    for tidx, c in enumerate(cfgs):
        logger.info(f"Trial {c.get('trial_id')} "
                    f": vae_lr={c.get('vae_lr'):.6f}, latent_dim={c.get('latent_dim')}, batch_size={c.get('batch_size')}")

    # Count how many keys actually vary
    varying_keys = [k for k, n in unique_counts.items() if n > 1]
    if len(varying_keys) == 0:
        logger.warning("⚠️  Only 1/5 unique configurations")
        for k in keys_to_check:
            logger.info(f"Parameter {k}: {unique_counts[k]} unique values = {uniq[k]}")
        logger.error("- Detected configs likely loaded from the same file due to fuzzy filename matching.")
        logger.error("- Ensure filenames match exactly (e.g., config_9.json should NOT resolve to config_69.json).")
        return False
    else:
        logger.info(f"✅ PASS: Configs show variation across keys: {varying_keys}")
        return True


def test_model_initialization_variation() -> bool:
    sec("TEST 3: Model Initialization Variation")
    logger.info("Testing model initialization variation...")

    # Keep models un-compiled during this test to simplify inspection (can be overridden by env)
    os.environ.setdefault("COMPILE_MODELS", "0")

    try:
        import main as mainmod
    except Exception as e:
        logger.error(f"❌ FAIL: Could not import main.py: {e}")
        return False

    # Use a typical observation shape seen in your dataset: (num_assets, num_features) = (30, 25)
    obs_shape = (30, 25)

    # Two different seeds -> two different initial weights (fingerprints)
    try:
        # Seed A
        torch.manual_seed(0)
        np.random.seed(0)
        cfgA = mainmod.StudyConfig(trial_id=0, seed=0, exp_name="smoke_modelA")
        cfgA.num_assets = 30
        vaeA, polA = mainmod.initialize_models(cfgA, obs_shape)
        fA_v = fingerprint_model(vaeA)
        fA_p = fingerprint_model(polA)

        # Seed B
        torch.manual_seed(1)
        np.random.seed(1)
        cfgB = mainmod.StudyConfig(trial_id=0, seed=1, exp_name="smoke_modelB")
        cfgB.num_assets = 30
        vaeB, polB = mainmod.initialize_models(cfgB, obs_shape)
        fB_v = fingerprint_model(vaeB)
        fB_p = fingerprint_model(polB)
    except Exception as e:
        logger.error(f"❌ FAIL: Model initialization test failed: {e}")
        return False

    def _diff(f1, f2):
        return any(abs(f1[k] - f2[k]) > 1e-7 for k in ("sum", "mean", "std"))

    vae_diff = _diff(fA_v, fB_v)
    pol_diff = _diff(fA_p, fB_p)

    if vae_diff and pol_diff:
        logger.info("✅ PASS: Different seeds lead to different initial model weights")
        return True
    else:
        logger.error("❌ FAIL: Model weights look too similar across seeds")
        logger.error(f"  VAE A: {fA_v} | VAE B: {fB_v}")
        logger.error(f"  POL A: {fA_p} | POL B: {fB_p}")
        return False


def test_training_variation(quick: bool) -> bool:
    sec("TEST 4: Training Variation")
    logger.info("Running minimal training test (20 episodes)..." if quick else "Running short training test...")

    # Keep models un-compiled during this test as well (reduce variance, faster startup)
    os.environ.setdefault("COMPILE_MODELS", "0")

    # Patch torch.as_tensor to ignore unexpected non_blocking kwargs from training code
    patch_torch_as_tensor()

    try:
        import main as mainmod
    except Exception as e:
        logger.error(f"❌ FAIL: Could not import main.py: {e}")
        return False

    try:
        logger.info("Testing training variation...")
        if quick:
            logger.info("Quick mode: Testing with minimal training episodes")
        # Prepare a minimal config
        run_config = mainmod.StudyConfig(trial_id=69, seed=42, exp_name="smoke_quick")
        # Keep it small and deterministic-ish
        run_config.max_episodes = 20 if quick else 50
        run_config.episodes_per_task = 1
        run_config.val_interval = 10 ** 9  # effectively disable mid-training validation
        run_config.num_envs = 1            # avoid heavier batched env path for the test

        # Prepare data once
        split_tensors, _ = mainmod.prepare_datasets(run_config)

        start = time.time()
        result = mainmod.train_single_run(run_config, split_tensors)
        dur = time.time() - start

        if not isinstance(result, dict) or "episodes_trained" not in result:
            logger.error("❌ FAIL: Training result is malformed")
            return False

        logger.info(f"Training finished in {dur:.2f}s, episodes_trained={result['episodes_trained']}")
        logger.info("✅ PASS: Training ran to completion and produced a result")
        return True

    except Exception as e:
        logger.error(f"❌ FAIL: Training variation test failed: {e}", exc_info=True)
        return False


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Final Confirmatory Study - Smoke Test")
    parser.add_argument("--quick", action="store_true", help="Run the quick version of the smoke test")
    parser.add_argument("--configs-dir", type=str, default="experiment_configs")
    args = parser.parse_args()

    banner("FINAL STUDY SMOKE TEST")
    logger.info(f"Mode: {'Quick' if args.quick else 'Full'}")

    results = []

    sec("TEST 1: Seed Isolation")
    ok1 = test_seed_isolation()
    results.append(("Seed Isolation", ok1))

    ok2 = test_config_loading(args.configs_dir)
    results.append(("Config Loading", ok2))

    ok3 = test_model_initialization_variation()
    results.append(("Model Initialization", ok3))

    ok4 = test_training_variation(args.quick)
    results.append(("Training Variation", ok4))

    # Summary
    banner("SMOKE TEST SUMMARY")
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        if ok:
            logger.info(f"✅ PASS {name}")
        else:
            logger.info(f"❌ FAIL {name}")

    logger.info(f"\nOverall: {passed}/{len(results)} tests passed")
    if passed != len(results):
        logger.error("🚨 SOME TESTS FAILED - Fix issues before running full study")
        logger.error("   The main study will likely produce invalid results")

    sec("RECOMMENDATIONS")
    if not ok2:
        logger.error("- Check experiment_configs/ directory - may be missing exact files or using fuzzy matches")
    if not ok3:
        logger.error("- Model initialization not varying with seeds - ensure seeds are set BEFORE model construction")
    if not ok4:
        logger.error("- Training failed - if the error mentions 'non_blocking', keep this smoke test in place "
                     "or remove 'non_blocking' from torch.as_tensor calls in trainer/evaluator code")

    # Exit nonzero if any failures (useful for CI)
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
