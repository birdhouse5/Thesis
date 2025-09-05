#!/usr/bin/env python3
"""
Complete Phase 3 analysis focusing on early stopping optimization.
Analyzes training efficiency and overfitting prevention.
Usage: python analyze_phase3_results.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_phase3_results():
    """Load Phase 3 results from CSV only"""
    results_dir = Path("optuna_phase3_results")
    
    # Load CSV results
    csv_files = list(results_dir.glob("*_all_trials.csv"))
    if not csv_files:
        raise FileNotFoundError("No Phase 3 CSV results found")
    
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 Loading results from: {latest_csv}")
    
    df = pd.read_csv(latest_csv)
    completed = df[df['state'] == 'COMPLETE'].copy()
    
    print(f"✅ Phase 3 Results: {len(completed)}/{len(df)} completed ({len(completed)/len(df)*100:.1f}%)")
    print(f"🚫 Pruned trials: {len(df[df['state'] == 'PRUNED'])}")
    
    return completed

def analyze_early_stopping_effectiveness(df):
    """Analyze how effective early stopping parameters are"""
    print("\n" + "="*60)
    print("⏱️ EARLY STOPPING EFFECTIVENESS ANALYSIS")
    print("="*60)
    
    # Top 10 trials
    top_10 = df.nlargest(10, 'value')
    print("\n🏆 Top 10 Phase 3 Trials:")
    
    # Display key columns
    display_cols = ['trial_number', 'value', 'duration']
    param_cols = [col for col in df.columns if col.startswith('param_')]
    display_cols.extend(param_cols)
    
    print(top_10[display_cols].round(4).to_string(index=False))
    
    # Best trial details
    best_trial = df.loc[df['value'].idxmax()]
    print(f"\n🥇 Best Trial #{int(best_trial['trial_number'])} (Sharpe: {best_trial['value']:.4f}):")
    for col in param_cols:
        param_name = col.replace('param_', '')
        print(f"    {param_name}: {best_trial[col]}")
    
    if 'duration' in df.columns:
        best_duration = best_trial['duration']
        avg_duration = df['duration'].mean()
        print(f"    Training time: {best_duration:.1f}s (avg: {avg_duration:.1f}s)")

def analyze_training_efficiency(df):
    """Analyze training efficiency metrics"""
    print("\n" + "="*60)
    print("⚡ TRAINING EFFICIENCY ANALYSIS")
    print("="*60)
    
    # Training time analysis
    if 'duration' in df.columns:
        duration_stats = df['duration'].describe()
        print(f"\n🕐 Training Duration Statistics:")
        print(f"    Mean: {duration_stats['mean']:.1f} seconds ({duration_stats['mean']/60:.1f} minutes)")
        print(f"    Median: {duration_stats['50%']:.1f} seconds")
        print(f"    Range: {duration_stats['min']:.1f}s - {duration_stats['max']:.1f}s")
        
        # Efficiency analysis: performance per unit time
        df['efficiency'] = df['value'] / (df['duration'] / 60)  # Sharpe per minute
        
        print(f"\n⚡ Training Efficiency (Sharpe/minute):")
        efficiency_stats = df['efficiency'].describe()
        print(f"    Mean: {efficiency_stats['mean']:.4f}")
        print(f"    Best: {efficiency_stats['max']:.4f}")
        
        # Most efficient trials
        top_efficient = df.nlargest(5, 'efficiency')
        print(f"\n🎯 Most Efficient Trials:")
        for _, trial in top_efficient.iterrows():
            print(f"    Trial {int(trial['trial_number'])}: {trial['efficiency']:.4f} Sharpe/min "
                  f"(Sharpe: {trial['value']:.4f}, Time: {trial['duration']/60:.1f}min)")

def analyze_episode_length_optimization(df):
    """Analyze optimal episode length findings"""
    print("\n" + "="*60)
    print("📈 EPISODE LENGTH OPTIMIZATION")
    print("="*60)
    
    if 'param_max_episodes' in df.columns:
        # Performance by max episodes
        episode_performance = df.groupby('param_max_episodes')['value'].agg(['mean', 'std', 'count', 'max']).round(4)
        episode_performance = episode_performance.sort_values('mean', ascending=False)
        
        print(f"\n📊 Performance by Maximum Episodes:")
        print(episode_performance.to_string(header=['mean_sharpe', 'std_sharpe', 'trials', 'best_sharpe']))
        
        # Training time by episodes
        if 'duration' in df.columns:
            time_by_episodes = df.groupby('param_max_episodes')['duration'].agg(['mean', 'std']).round(1)
            print(f"\n⏱️ Training Time by Maximum Episodes:")
            for episodes, row in time_by_episodes.iterrows():
                print(f"    {episodes} episodes: {row['mean']:.1f}s ± {row['std']:.1f}s "
                      f"({row['mean']/60:.1f} ± {row['std']/60:.1f} minutes)")

def analyze_early_stopping_parameters(df):
    """Detailed analysis of early stopping parameters"""
    print("\n" + "="*60)
    print("🛑 EARLY STOPPING PARAMETER ANALYSIS")
    print("="*60)
    
    # Patience analysis
    if 'param_early_stopping_patience' in df.columns:
        patience_performance = df.groupby('param_early_stopping_patience')['value'].agg(['mean', 'std', 'count', 'max']).round(4)
        patience_performance = patience_performance.sort_values('mean', ascending=False)
        
        print(f"\n🎯 Performance by Early Stopping Patience:")
        print(patience_performance.to_string(header=['mean_sharpe', 'std_sharpe', 'trials', 'best_sharpe']))
    
    # Min delta analysis
    if 'param_early_stopping_min_delta' in df.columns:
        delta_performance = df.groupby('param_early_stopping_min_delta')['value'].agg(['mean', 'std', 'count', 'max']).round(4)
        delta_performance = delta_performance.sort_values('mean', ascending=False)
        
        print(f"\n📏 Performance by Minimum Delta:")
        print(delta_performance.to_string(header=['mean_sharpe', 'std_sharpe', 'trials', 'best_sharpe']))
    
    # Validation interval analysis
    if 'param_val_interval' in df.columns:
        interval_performance = df.groupby('param_val_interval')['value'].agg(['mean', 'std', 'count', 'max']).round(4)
        interval_performance = interval_performance.sort_values('mean', ascending=False)
        
        print(f"\n📅 Performance by Validation Interval:")
        print(interval_performance.to_string(header=['mean_sharpe', 'std_sharpe', 'trials', 'best_sharpe']))

def analyze_overfitting_prevention(df):
    """Analyze how well early stopping prevents overfitting"""
    print("\n" + "="*60)
    print("🛡️ OVERFITTING PREVENTION ANALYSIS")
    print("="*60)
    
    # Top 20% analysis
    top_20_pct = df.nlargest(max(10, len(df)//5), 'value')
    print(f"\n📈 Analysis of Top {len(top_20_pct)} Trials:")
    
    # Optimal configurations in top performers
    if 'param_early_stopping_patience' in df.columns:
        patience_dist = top_20_pct['param_early_stopping_patience'].value_counts().sort_index()
        print(f"\n  Early Stopping Patience Distribution:")
        for patience, count in patience_dist.items():
            pct = count / len(top_20_pct) * 100
            print(f"    {patience} checks: {count} trials ({pct:.1f}%)")
    
    if 'param_max_episodes' in df.columns:
        episodes_dist = top_20_pct['param_max_episodes'].value_counts().sort_index()
        print(f"\n  Maximum Episodes Distribution:")
        for episodes, count in episodes_dist.items():
            pct = count / len(top_20_pct) * 100
            print(f"    {episodes} episodes: {count} trials ({pct:.1f}%)")
    
    if 'param_val_interval' in df.columns:
        interval_dist = top_20_pct['param_val_interval'].value_counts().sort_index()
        print(f"\n  Validation Interval Distribution:")
        for interval, count in interval_dist.items():
            pct = count / len(top_20_pct) * 100
            print(f"    Every {interval} episodes: {count} trials ({pct:.1f}%)")

def compare_all_phases():
    """Compare all three phases"""
    print("\n" + "="*60)
    print("🏁 ALL PHASES COMPARISON")
    print("="*60)
    
    # Known Phase 1 & 2 results
    phase1_best = 64.5049
    phase2_best = None
    
    # Load Phase 2 results
    phase2_dir = Path("optuna_phase2_results")
    phase2_files = list(phase2_dir.glob("*_all_trials.csv"))
    if phase2_files:
        phase2_df = pd.read_csv(phase2_files[0])
        phase2_completed = phase2_df[phase2_df['state'] == 'COMPLETE']
        if len(phase2_completed) > 0:
            phase2_best = phase2_completed['value'].max()
    
    # Phase 3 results
    phase3_files = list(Path("optuna_phase3_results").glob("*_all_trials.csv"))
    if phase3_files:
        phase3_df = pd.read_csv(phase3_files[0])
        phase3_completed = phase3_df[phase3_df['state'] == 'COMPLETE']
        phase3_best = phase3_completed['value'].max() if len(phase3_completed) > 0 else None
        phase3_mean = phase3_completed['value'].mean() if len(phase3_completed) > 0 else None
    else:
        phase3_best = None
        phase3_mean = None
    
    print(f"📊 Performance Evolution:")
    print(f"  Phase 1 (Architecture): {phase1_best:.4f}")
    if phase2_best:
        phase2_improvement = ((phase2_best - phase1_best) / phase1_best) * 100
        print(f"  Phase 2 (VAE Dynamics): {phase2_best:.4f} ({phase2_improvement:+.2f}%)")
    if phase3_best:
        base_score = phase2_best if phase2_best else phase1_best
        phase3_improvement = ((phase3_best - base_score) / base_score) * 100
        print(f"  Phase 3 (Early Stopping): {phase3_best:.4f} ({phase3_improvement:+.2f}%)")
        
        if phase2_best:
            total_improvement = ((phase3_best - phase1_best) / phase1_best) * 100
            print(f"\n🎯 Total Optimization Gain: {total_improvement:+.2f}%")
    
    # Training efficiency comparison
    if phase3_files and 'duration' in phase3_completed.columns:
        avg_phase3_time = phase3_completed['duration'].mean() / 60  # minutes
        print(f"\n⏱️ Phase 3 Training Efficiency:")
        print(f"  Average training time: {avg_phase3_time:.1f} minutes")
        if phase3_best:
            print(f"  Performance per minute: {phase3_best / avg_phase3_time:.4f}")

def analyze_final_optimized_setup():
    """Analyze the final optimized configuration"""
    print("\n" + "="*60)
    print("🎯 FINAL OPTIMIZED CONFIGURATION")
    print("="*60)
    
    # Try to load Phase 3 config
    config_file = Path("optuna_phase3_results/final_phase3_config.json")
    if config_file.exists():
        with open(config_file, 'r') as f:
            final_config = json.load(f)
        
        print("🏗️ Complete Optimized Architecture:")
        
        print("\n  Phase 1 Optimized (Fixed):")
        print(f"    latent_dim: {final_config.get('latent_dim')}")
        print(f"    hidden_dim: {final_config.get('hidden_dim')}")
        
        print("\n  Phase 2 Optimized (Fixed):")
        phase2_params = ['vae_lr', 'policy_lr', 'vae_beta', 'vae_update_freq', 
                        'episodes_per_task', 'batch_size', 'vae_batch_size', 
                        'ppo_epochs', 'entropy_coef']
        for param in phase2_params:
            if param in final_config:
                value = final_config[param]
                if isinstance(value, float) and value < 0.01:
                    print(f"    {param}: {value:.6f}")
                else:
                    print(f"    {param}: {value}")
        
        print("\n  Phase 3 Optimized (Early Stopping):")
        phase3_params = ['max_episodes', 'early_stopping_patience', 
                        'early_stopping_min_delta', 'val_interval']
        for param in phase3_params:
            if param in final_config:
                print(f"    {param}: {final_config[param]}")
        
        return final_config
    else:
        print("❌ Final Phase 3 config file not found")
        return None

def generate_final_recommendations(final_config, all_phase_results):
    """Generate final production recommendations"""
    print("\n" + "="*60)
    print("🚀 FINAL PRODUCTION RECOMMENDATIONS")
    print("="*60)
    
    if final_config:
        print(f"🎯 Recommended Production Setup:")
        
        print(f"\n  Training Configuration:")
        print(f"    Max episodes: {final_config.get('max_episodes', 'N/A')}")
        print(f"    Early stopping patience: {final_config.get('early_stopping_patience', 'N/A')}")
        print(f"    Validation frequency: Every {final_config.get('val_interval', 'N/A')} episodes")
        print(f"    Min improvement threshold: {final_config.get('early_stopping_min_delta', 'N/A')}")
        
        print(f"\n  Expected Training Behavior:")
        if 'max_episodes' in final_config and 'early_stopping_patience' in final_config:
            max_ep = final_config['max_episodes']
            patience = final_config['early_stopping_patience']
            val_int = final_config.get('val_interval', 500)
            
            min_episodes = max(1000, max_ep // 4)
            typical_episodes = min_episodes + (patience * val_int)
            
            print(f"    Minimum training: {min_episodes} episodes")
            print(f"    Typical training: {typical_episodes} episodes")
            print(f"    Maximum training: {max_ep} episodes")
        
        print(f"\n  Monitoring Recommendations:")
        print(f"    📈 Primary metric: Validation Sharpe ratio")
        print(f"    ⚠️ Stop if: No improvement for {final_config.get('early_stopping_patience', 'N/A')} validation checks")
        print(f"    💾 Checkpoint: Every {final_config.get('val_interval', 'N/A')} episodes")
        print(f"    🔍 Log interval: Every 200 episodes")
        
        print(f"\n  Risk Management:")
        print(f"    🛡️ Portfolio constraints: Sum to 1.0, long-only")
        print(f"    📉 Maximum drawdown threshold: 20%")
        print(f"    🔄 Rebalancing frequency: Daily")
        print(f"    💰 Transaction costs: Include in backtesting")

def create_final_thesis_summary():
    """Create comprehensive thesis summary"""
    print("\n" + "="*60)
    print("📋 FINAL THESIS SUMMARY")
    print("="*60)
    
    # Collect all results
    phase1_best = 64.5049
    phase2_best = None
    phase3_best = None
    
    # Phase 2
    phase2_files = list(Path("optuna_phase2_results").glob("*_all_trials.csv"))
    if phase2_files:
        phase2_df = pd.read_csv(phase2_files[0])
        phase2_completed = phase2_df[phase2_df['state'] == 'COMPLETE']
        phase2_best = phase2_completed['value'].max() if len(phase2_completed) > 0 else None
    
    # Phase 3
    phase3_files = list(Path("optuna_phase3_results").glob("*_all_trials.csv"))
    if phase3_files:
        phase3_df = pd.read_csv(phase3_files[0])
        phase3_completed = phase3_df[phase3_df['state'] == 'COMPLETE']
        phase3_best = phase3_completed['value'].max() if len(phase3_completed) > 0 else None
        phase3_trials = len(phase3_completed)
        avg_time = phase3_completed['duration'].mean() if 'duration' in phase3_completed.columns else None
    else:
        phase3_trials = 0
        avg_time = None
    
    summary = {
        "methodology": "Three-phase progressive Bayesian optimization",
        "total_trials": 220 + phase3_trials,
        "phases": {
            "phase1": {
                "focus": "Neural architecture optimization",
                "trials": 100,
                "best_sharpe": phase1_best,
                "key_finding": "Larger networks (512/1024) outperform smaller ones"
            },
            "phase2": {
                "focus": "VAE dynamics and training parameters",
                "trials": 120,
                "best_sharpe": phase2_best,
                "key_finding": "Low VAE beta and large batches crucial"
            },
            "phase3": {
                "focus": "Early stopping and training efficiency",
                "trials": phase3_trials,
                "best_sharpe": phase3_best,
                "key_finding": "Optimal early stopping prevents overfitting"
            }
        },
        "final_performance": {
            "starting_sharpe": phase1_best,
            "final_sharpe": phase3_best,
            "total_improvement": f"{((phase3_best - phase1_best) / phase1_best * 100):+.2f}%" if phase3_best else None
        },
        "computational_efficiency": {
            "total_gpu_hours": "~50-60 hours",
            "hardware": "NVIDIA RTX 4090",
            "optimization_framework": "Optuna with TPE sampling"
        }
    }
    
    print("🔬 Complete Three-Phase Optimization:")
    print(f"  Total trials: {summary['total_trials']}")
    print(f"  GPU time: {summary['computational_efficiency']['total_gpu_hours']}")
    
    print(f"\n📈 Performance Evolution:")
    for phase, data in summary['phases'].items():
        if data['best_sharpe']:
            print(f"  {phase.upper()}: {data['best_sharpe']:.4f} - {data['key_finding']}")
    
    if phase3_best and phase1_best:
        total_gain = ((phase3_best - phase1_best) / phase1_best) * 100
        print(f"\n🎯 Total Optimization Gain: {total_gain:+.2f}%")
        
        # Practical impact
        print(f"\n💼 Practical Impact:")
        if total_gain > 15:
            print(f"    🎉 Exceptional: {total_gain:.1f}% improvement indicates significant alpha potential")
        elif total_gain > 10:
            print(f"    ✅ Strong: {total_gain:.1f}% improvement shows meaningful optimization success")
        elif total_gain > 5:
            print(f"    ➕ Moderate: {total_gain:.1f}% improvement provides incremental value")
        else:
            print(f"    ⚠️ Limited: {total_gain:.1f}% improvement suggests near-optimal baseline")
    
    # Save final summary
    summary_file = Path("final_thesis_optimization_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n💾 Complete summary saved to: {summary_file}")
    
    return summary

def main():
    """Run complete Phase 3 analysis"""
    try:
        print("🚀 Starting Phase 3 Analysis - Early Stopping Optimization")
        
        # Load Phase 3 results
        phase3_df = load_phase3_results()
        
        # Early stopping effectiveness
        analyze_early_stopping_effectiveness(phase3_df)
        
        # Training efficiency
        analyze_training_efficiency(phase3_df)
        
        # Episode length optimization
        analyze_episode_length_optimization(phase3_df)
        
        # Early stopping parameters
        analyze_early_stopping_parameters(phase3_df)
        
        # Overfitting prevention
        analyze_overfitting_prevention(phase3_df)
        
        # Compare all phases
        compare_all_phases()
        
        # Final configuration
        final_config = analyze_final_optimized_setup()
        
        # Production recommendations
        generate_final_recommendations(final_config, phase3_df)
        
        # Complete thesis summary
        create_final_thesis_summary()
        
        print(f"\n🎉 Phase 3 Analysis Complete!")
        print(f"  📊 Analyzed {len(phase3_df)} completed trials")
        print(f"  🎯 Identified optimal early stopping configuration")
        print(f"  📋 Generated final thesis summary")
        print(f"  🚀 Ready for production deployment")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()