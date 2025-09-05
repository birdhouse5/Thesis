#!/usr/bin/env python3
"""
Validate Phase 2 results and check for potential issues.
Usage: python validate_phase2_results.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_best_trial_details():
    """Deep dive into the best trial to validate results"""
    print("🔍 VALIDATING BEST TRIAL RESULTS")
    print("="*60)
    
    # Look for trial 46 folder
    trial_dirs = list(Path("results/optuna_phase2_runs").glob("trial_46_*"))
    if not trial_dirs:
        print("❌ Trial 46 folder not found. Looking for alternative evidence...")
        return False
    
    trial_dir = trial_dirs[0]
    print(f"📁 Found trial 46: {trial_dir}")
    
    # Check for metrics file
    metrics_file = trial_dir / "metrics.csv"
    if metrics_file.exists():
        metrics_df = pd.read_csv(metrics_file)
        print(f"📈 Loaded {len(metrics_df)} training records")
        
        # Analyze validation progression
        val_metrics = metrics_df[metrics_df['split'] == 'val']
        if len(val_metrics) > 0:
            print(f"\n📊 Validation Progression:")
            print(f"  Validation checks: {len(val_metrics)}")
            print(f"  Final validation Sharpe: {val_metrics['value'].iloc[-1]:.4f}")
            print(f"  Max validation Sharpe: {val_metrics['value'].max():.4f}")
            print(f"  Validation trend: {'Improving' if val_metrics['value'].iloc[-1] > val_metrics['value'].iloc[0] else 'Declining'}")
            
            # Check for overfitting signs
            if len(val_metrics) >= 3:
                recent_trend = val_metrics['value'].iloc[-3:].is_monotonic_increasing
                print(f"  Recent trend: {'Still improving' if recent_trend else 'Potentially plateauing'}")
        
        # Analyze training progression
        train_metrics = metrics_df[metrics_df['split'] == 'train']
        if len(train_metrics) > 0:
            print(f"\n🏋️ Training Progression:")
            print(f"  Training episodes: {len(train_metrics)}")
            print(f"  Final training Sharpe: {train_metrics['value'].iloc[-1]:.4f}")
            print(f"  Training volatility: {train_metrics['value'].std():.4f}")
            
            # Check training stability
            if train_metrics['value'].std() > 50:
                print("  ⚠️  High training volatility - potential instability")
            else:
                print("  ✅ Stable training progression")
        
        return True
    else:
        print("❌ No metrics.csv found for trial 46")
        return False

def check_parameter_patterns():
    """Analyze the parameter patterns that led to success"""
    print("\n🎛️ ANALYZING SUCCESS PATTERNS")
    print("="*60)
    
    # Key insights from your results
    print("🔑 Key Success Factors Identified:")
    print("  • Sequence Length: 120 days (77.8% of top trials)")
    print("  • Batch Size: 8192 (88.9% of top trials)")
    print("  • Episodes per Task: 3 (50% of top trials)")
    print("  • VAE Update Freq: 5 (dominates performance)")
    print("  • PPO Epochs: 8 (preferred in top trials)")
    
    print("\n📈 Parameter Correlations:")
    print("  • seq_len: +0.225 (longer context helps)")
    print("  • vae_update_freq: +0.174 (frequent VAE updates)")
    print("  • batch_size: +0.193 (large batches better)")
    print("  • vae_lr: -0.220 (lower VAE LR preferred)")
    
    print("\n🧠 Interpretation:")
    print("  ✅ Long context (120 days) enables better task recognition")
    print("  ✅ Large batches (8192) provide stable gradients")
    print("  ✅ Frequent VAE updates (5x) improve task adaptation")
    print("  ✅ Few episodes per task (3) prevents overfitting to single regime")
    print("  ⚠️  Low VAE beta (0.0126) might indicate minimal regularization")

def assess_realism():
    """Assess if the results are realistic"""
    print("\n🎯 REALISM ASSESSMENT")
    print("="*60)
    
    sharpe_121 = 121.7145
    
    print(f"📊 Result Analysis:")
    print(f"  Best Sharpe: {sharpe_121:.4f}")
    print(f"  Daily Sharpe (if annualized): {sharpe_121/np.sqrt(252):.4f}")
    print(f"  Weekly Sharpe (if annualized): {sharpe_121/np.sqrt(52):.4f}")
    
    print(f"\n🌍 Real-World Context:")
    print(f"  • Renaissance Medallion Fund: ~39% annual return, ~1.5-2.0 Sharpe")
    print(f"  • Top quant funds: 15-25% annual return, ~1.0-1.5 Sharpe")
    print(f"  • S&P 500 (1990-2020): ~10% annual return, ~0.4-0.6 Sharpe")
    
    print(f"\n⚠️  Potential Issues:")
    if sharpe_121 > 10:
        print(f"  🚨 Extremely high Sharpe ratio suggests:")
        print(f"    • Possible data leakage or look-ahead bias")
        print(f"    • Overfitting to validation period")
        print(f"    • Unrealistic assumptions (no transaction costs)")
        print(f"    • Bug in reward calculation")
    
    print(f"\n🔍 Validation Checklist:")
    print(f"  □ Check temporal split integrity")
    print(f"  □ Verify no future information leakage")
    print(f"  □ Test on completely held-out data")
    print(f"  □ Add transaction costs to reward")
    print(f"  □ Compare against simple baselines")

def generate_validation_plan():
    """Generate a comprehensive validation plan"""
    print("\n📋 VALIDATION PLAN")
    print("="*60)
    
    print("🔬 Immediate Tests (Next 1-2 days):")
    print("  1. Test best model on held-out test set (2021-2024)")
    print("  2. Implement transaction cost penalties (0.1-0.5% per trade)")
    print("  3. Compare against equal-weight portfolio baseline")
    print("  4. Analyze portfolio allocations for reasonableness")
    print("  5. Check for excessive turnover or concentration")
    
    print("\n🛡️ Robustness Tests (Next 1 week):")
    print("  6. Walk-forward validation on different time periods")
    print("  7. Stress test on 2008 financial crisis period")
    print("  8. Sensitivity analysis to hyperparameters")
    print("  9. Bootstrap validation with multiple random seeds")
    print("  10. Out-of-sample test on different asset universes")
    
    print("\n📊 Baseline Comparisons:")
    baselines = [
        ("Equal Weight", "Simple 1/N allocation"),
        ("Market Cap", "SPY index fund"),
        ("60/40", "Traditional stock/bond mix"),
        ("Buy & Hold", "SPY buy and hold"),
        ("Moving Average", "Simple technical strategy")
    ]
    
    for name, desc in baselines:
        print(f"  • {name}: {desc}")
    
    print("\n🎯 Success Criteria:")
    print("  • Test Sharpe > 1.0 (excellent)")
    print("  • Max drawdown < 30%")
    print("  • Beats equal-weight by >2% annually")
    print("  • Portfolio turnover < 300% annually")
    print("  • Consistent across different market regimes")

def main():
    """Run comprehensive validation analysis"""
    print("🔬 COMPREHENSIVE PHASE 2 VALIDATION")
    print("="*80)
    
    # Analyze trial details if available
    found_details = analyze_best_trial_details()
    
    # Check parameter patterns
    check_parameter_patterns()
    
    # Assess realism
    assess_realism()
    
    # Generate validation plan
    generate_validation_plan()
    
    print("\n" + "="*80)
    print("🎯 IMMEDIATE NEXT STEPS")
    print("="*80)
    
    if found_details:
        print("✅ Found trial details - proceed with model testing")
    else:
        print("⚠️  Limited trial details - focus on config validation")
    
    print("\n🚀 Recommended Actions:")
    print("  1. Run final model on test set (2021-2024)")
    print("  2. Implement transaction costs")
    print("  3. Compare vs simple baselines")
    print("  4. If results hold: proceed to production training")
    print("  5. If results don't hold: investigate and debug")
    
    print("\n📝 For Thesis:")
    print("  • Document the 88.69% improvement")
    print("  • Include robustness testing results")
    print("  • Discuss realism and limitations")
    print("  • Compare against established benchmarks")

if __name__ == "__main__":
    main()