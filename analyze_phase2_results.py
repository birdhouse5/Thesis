#!/usr/bin/env python3
"""
Complete Phase 2 analysis without requiring Optuna study object.
Works with CSV results only to avoid version compatibility issues.
Usage: python analyze_phase2_results_fixed.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_phase2_results():
    """Load Phase 2 results from CSV only"""
    results_dir = Path("experiments/results/optuna_phase2_results")
    
    # Load CSV results
    csv_files = list(results_dir.glob("*_all_trials.csv"))
    if not csv_files:
        raise FileNotFoundError("No Phase 2 CSV results found")
    
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 Loading results from: {latest_csv}")
    
    df = pd.read_csv(latest_csv)
    completed = df[df['state'] == 'COMPLETE'].copy()
    
    print(f"✅ Phase 2 Results: {len(completed)}/{len(df)} completed ({len(completed)/len(df)*100:.1f}%)")
    print(f"🚫 Pruned trials: {len(df[df['state'] == 'PRUNED'])}")
    
    return completed

def analyze_phase2_parameters(df):
    """Analyze which Phase 2 parameters matter most"""
    print("\n" + "="*60)
    print("🎛️  PHASE 2 PARAMETER ANALYSIS")
    print("="*60)
    
    # Top 10 trials
    top_10 = df.nlargest(10, 'value')
    print("\n🏆 Top 10 Phase 2 Trials:")
    
    # Display key columns nicely
    display_cols = ['trial_number', 'value']
    param_cols = [col for col in df.columns if col.startswith('param_')]
    display_cols.extend(param_cols)
    
    print(top_10[display_cols].round(4).to_string(index=False))
    
    # Best trial details
    best_trial = df.loc[df['value'].idxmax()]
    print(f"\n🥇 Best Trial #{int(best_trial['trial_number'])} (Sharpe: {best_trial['value']:.4f}):")
    for col in param_cols:
        param_name = col.replace('param_', '')
        print(f"    {param_name}: {best_trial[col]}")
    
    # Parameter importance analysis
    print(f"\n📊 Parameter Performance Analysis:")
    
    for param in param_cols:
        param_name = param.replace('param_', '')
        
        if df[param].dtype in ['int64', 'float64']:
            # Check if we have enough unique values for quartile analysis
            unique_values = df[param].nunique()
            
            if unique_values >= 4:
                # Continuous parameter with enough variation - correlation analysis
                correlation = df[param].corr(df['value'])
                print(f"\n  {param_name} (continuous):")
                print(f"    Correlation with performance: {correlation:.3f}")
                
                # Quartile analysis with duplicate handling
                try:
                    quartiles = pd.qcut(df[param], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
                    quartile_performance = df.groupby(quartiles)['value'].agg(['mean', 'std', 'count']).round(3)
                    print("    Performance by quartile:")
                    print(quartile_performance.to_string(header=['    mean_perf', 'std_perf', 'count']))
                except ValueError:
                    # Fallback to treating as categorical if quartiles still fail
                    print(f"    (Treating as categorical due to limited unique values: {unique_values})")
                    param_performance = df.groupby(param)['value'].agg(['mean', 'std', 'count']).round(3)
                    param_performance = param_performance.sort_values('mean', ascending=False)
                    print(param_performance.to_string(header=['    mean_perf', 'std_perf', 'count']))
            else:
                # Too few unique values - treat as categorical
                correlation = df[param].corr(df['value'])
                print(f"\n  {param_name} (discrete, {unique_values} values):")
                print(f"    Correlation with performance: {correlation:.3f}")
                param_performance = df.groupby(param)['value'].agg(['mean', 'std', 'count']).round(3)
                param_performance = param_performance.sort_values('mean', ascending=False)
                print(param_performance.to_string(header=['    mean_perf', 'std_perf', 'count']))
        else:
            # Categorical parameter
            param_performance = df.groupby(param)['value'].agg(['mean', 'std', 'count']).round(3)
            param_performance = param_performance.sort_values('mean', ascending=False)
            print(f"\n  {param_name} (categorical):")
            print(param_performance.to_string(header=['    mean_perf', 'std_perf', 'count']))

def analyze_key_phase2_insights(df):
    """Analyze specific Phase 2 insights"""
    print("\n" + "="*60)
    print("🔍 KEY PHASE 2 INSIGHTS")
    print("="*60)
    
    # Top 20% analysis
    top_20_pct = df.nlargest(max(15, len(df)//5), 'value')
    print(f"\n📈 Analysis of Top {len(top_20_pct)} Trials:")
    
    # VAE Beta analysis
    if 'param_vae_beta' in df.columns:
        beta_stats = top_20_pct['param_vae_beta'].describe()
        print(f"\n  VAE Beta in top performers:")
        print(f"    Range: {beta_stats['min']:.4f} - {beta_stats['max']:.4f}")
        print(f"    Median: {beta_stats['50%']:.4f}")
        print(f"    Mean: {beta_stats['mean']:.4f}")
    
    # Sequence length analysis
    if 'param_seq_len' in df.columns:
        seq_counts = top_20_pct['param_seq_len'].value_counts()
        print(f"\n  Sequence lengths in top performers:")
        for seq_len, count in seq_counts.items():
            pct = count / len(top_20_pct) * 100
            print(f"    {seq_len} days: {count} trials ({pct:.1f}%)")
    
    # Batch size analysis
    if 'param_batch_size' in df.columns:
        batch_counts = top_20_pct['param_batch_size'].value_counts()
        print(f"\n  Batch sizes in top performers:")
        for batch_size, count in batch_counts.items():
            pct = count / len(top_20_pct) * 100
            print(f"    {batch_size}: {count} trials ({pct:.1f}%)")
    
    # Episodes per task analysis
    if 'param_episodes_per_task' in df.columns:
        ept_counts = top_20_pct['param_episodes_per_task'].value_counts()
        print(f"\n  Episodes per task in top performers:")
        for ept, count in ept_counts.items():
            pct = count / len(top_20_pct) * 100
            print(f"    {ept}: {count} trials ({pct:.1f}%)")

def compare_phase1_vs_phase2():
    """Compare Phase 1 and Phase 2 best results"""
    print("\n" + "="*60)
    print("🥇 PHASE 1 vs PHASE 2 COMPARISON")
    print("="*60)
    
    # Phase 1 best (from your known results)
    phase1_best = 64.5049
    print(f"Phase 1 Best Sharpe: {phase1_best:.4f}")
    print("  Architecture: latent_dim=512, hidden_dim=1024")
    print("  VAE LR: 0.002617, Policy LR: 0.001356")
    
    # Phase 2 results
    phase2_dir = Path("optuna_phase2_results")
    phase2_files = list(phase2_dir.glob("*_all_trials.csv"))
    if phase2_files:
        phase2_df = pd.read_csv(phase2_files[0])
        phase2_completed = phase2_df[phase2_df['state'] == 'COMPLETE']
        
        if len(phase2_completed) > 0:
            phase2_best = phase2_completed['value'].max()
            phase2_mean = phase2_completed['value'].mean()
            phase2_std = phase2_completed['value'].std()
            
            print(f"\nPhase 2 Results:")
            print(f"  Best Sharpe: {phase2_best:.4f}")
            print(f"  Mean Sharpe: {phase2_mean:.4f} ± {phase2_std:.4f}")
            print(f"  Trials completed: {len(phase2_completed)}")
            
            improvement = ((phase2_best - phase1_best) / phase1_best) * 100
            print(f"\n📊 Performance Comparison:")
            print(f"  Absolute improvement: {phase2_best - phase1_best:+.4f}")
            print(f"  Relative improvement: {improvement:+.2f}%")
            
            if improvement > 10:
                print("🎉 Major improvement! Phase 2 optimization was highly successful.")
            elif improvement > 5:
                print("✅ Significant improvement! Phase 2 provided meaningful gains.")
            elif improvement > 0:
                print("➕ Modest improvement. Phase 2 provided incremental gains.")
            else:
                print("⚠️  No improvement. Phase 1 may have been near-optimal.")
                
            return phase2_completed, phase2_best
        else:
            print("❌ No completed Phase 2 trials found")
            return None, None
    else:
        print("❌ No Phase 2 results found")
        return None, None

def analyze_final_config():
    """Analyze the final optimized configuration"""
    print("\n" + "="*60)
    print("🎯 FINAL OPTIMIZED CONFIGURATION")
    print("="*60)
    
    config_file = Path("optuna_phase2_results/final_optimized_config.json")
    if config_file.exists():
        with open(config_file, 'r') as f:
            final_config = json.load(f)
        
        print("🏗️  Fixed Architecture (from Phase 1):")
        print(f"  latent_dim: {final_config.get('latent_dim')}")
        print(f"  hidden_dim: {final_config.get('hidden_dim')}")
        
        print("\n📈 Refined Learning Rates:")
        print(f"  vae_lr: {final_config.get('vae_lr'):.6f}")
        print(f"  policy_lr: {final_config.get('policy_lr'):.6f}")
        
        print("\n🎛️  Optimized VAE Parameters:")
        vae_params = ['vae_beta', 'vae_update_freq', 'seq_len']
        for param in vae_params:
            if param in final_config:
                print(f"  {param}: {final_config[param]}")
        
        print("\n⚙️  Optimized Training Parameters:")
        training_params = ['batch_size', 'vae_batch_size', 'episodes_per_task', 'ppo_epochs', 'entropy_coef']
        for param in training_params:
            if param in final_config:
                print(f"  {param}: {final_config[param]}")
        
        return final_config
    else:
        print("❌ Final config file not found")
        return None

def generate_production_recommendations(final_config, phase2_best):
    """Generate recommendations for production runs"""
    print("\n" + "="*60)
    print("🚀 PRODUCTION RUN RECOMMENDATIONS")
    print("="*60)
    
    if final_config and phase2_best:
        print(f"🎯 Expected Performance:")
        print(f"  Validation Sharpe Ratio: {phase2_best:.4f}")
        
        # Estimate annualized performance
        if phase2_best > 0:
            # Assuming daily returns, convert to annualized
            daily_sharpe = phase2_best
            annual_sharpe = daily_sharpe * np.sqrt(252)  # 252 trading days
            print(f"  Estimated Annual Sharpe: {annual_sharpe:.2f}")
            
            # Estimate returns assuming market volatility
            market_vol = 0.16  # ~16% annual volatility
            estimated_return = annual_sharpe * market_vol
            print(f"  Estimated Annual Return: {estimated_return:.1%} (assuming 16% volatility)")
        
        print(f"\n⚙️  Production Training Settings:")
        print(f"  Recommended episodes: 10,000-15,000")
        print(f"  Validation frequency: Every 1,000 episodes")
        print(f"  Early stopping patience: 5-8 validation checks")
        print(f"  Checkpointing: Every 2,000 episodes")
        
        print(f"\n📊 Key Monitoring Metrics:")
        print(f"  Primary: Validation Sharpe ratio > {phase2_best-5:.1f}")
        print(f"  Risk: Maximum drawdown < 20%")
        print(f"  Efficiency: Portfolio turnover < 200% annually")
        print(f"  Training: VAE loss convergence, policy entropy > 0.1")
        
        print(f"\n🎮 Baseline Comparisons to Include:")
        print(f"  • Equal-weight portfolio (1/N strategy)")
        print(f"  • Market cap weighted (SPY benchmark)")
        print(f"  • 60/40 stock/bond portfolio")
        print(f"  • Buy-and-hold S&P 500")
        print(f"  • Simple momentum strategy")
        
    else:
        print("❌ Cannot generate recommendations without final config and results")

def create_thesis_summary():
    """Create summary data for thesis"""
    print("\n" + "="*60)
    print("📋 THESIS RESULTS SUMMARY")
    print("="*60)
    
    # Try to load both phase results
    phase1_best = 64.5049
    
    phase2_files = list(Path("optuna_phase2_results").glob("*_all_trials.csv"))
    if phase2_files:
        phase2_df = pd.read_csv(phase2_files[0])
        phase2_completed = phase2_df[phase2_df['state'] == 'COMPLETE']
        phase2_best = phase2_completed['value'].max() if len(phase2_completed) > 0 else None
    else:
        phase2_best = None
    
    summary = {
        "methodology": "Two-phase Bayesian hyperparameter optimization",
        "phase1": {
            "focus": "Neural architecture and learning rates",
            "trials": 100,
            "completion_rate": "74%",
            "best_sharpe": phase1_best,
            "best_architecture": "latent_dim=512, hidden_dim=1024"
        },
        "phase2": {
            "focus": "VAE dynamics and training parameters", 
            "trials": 120,
            "completion_rate": "77.5%",
            "best_sharpe": phase2_best,
            "improvement": f"{((phase2_best - phase1_best) / phase1_best * 100):+.2f}%" if phase2_best else None
        },
        "computational_resources": {
            "hardware": "NVIDIA RTX 4090 (24GB VRAM)",
            "total_time": "~35-45 hours",
            "total_trials": 220,
            "successful_trials": "~150"
        }
    }
    
    print("🔬 Complete Methodology Summary:")
    print(f"  Approach: {summary['methodology']}")
    print(f"  Total trials: {summary['computational_resources']['total_trials']}")
    print(f"  Success rate: ~{summary['computational_resources']['successful_trials']} trials")
    print(f"  Computational time: {summary['computational_resources']['total_time']}")
    
    print(f"\n📈 Performance Results:")
    print(f"  Phase 1 best: {summary['phase1']['best_sharpe']:.4f}")
    if phase2_best:
        print(f"  Phase 2 best: {summary['phase2']['best_sharpe']:.4f}")
        print(f"  Total improvement: {summary['phase2']['improvement']}")
    
    # Save summary
    summary_file = Path("thesis_optimization_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n💾 Summary saved to: {summary_file}")
    
    return summary

def main():
    """Run complete Phase 2 analysis"""
    try:
        print("🚀 Starting Phase 2 Analysis (CSV-only version)")
        
        # Load results
        phase2_df = load_phase2_results()
        
        # Analyze Phase 2 parameters
        analyze_phase2_parameters(phase2_df)
        
        # Key insights
        analyze_key_phase2_insights(phase2_df)
        
        # Compare phases
        phase2_completed, phase2_best = compare_phase1_vs_phase2()
        
        # Analyze final config
        final_config = analyze_final_config()
        
        # Production recommendations
        generate_production_recommendations(final_config, phase2_best)
        
        # Create thesis summary
        create_thesis_summary()
        
        print(f"\n🎉 Analysis complete! Key takeaways:")
        print(f"  📊 Phase 2 completed {len(phase2_df)} trials successfully")
        if phase2_best:
            print(f"  🏆 Best performance: {phase2_best:.4f} Sharpe ratio")
        print(f"  📋 Thesis summary: saved to thesis_optimization_summary.json")
        print(f"  🎯 Ready for production training with optimized config")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()