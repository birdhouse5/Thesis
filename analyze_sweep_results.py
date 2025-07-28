#!/usr/bin/env python3
"""
Analyze and compare parameter sweep results
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns

def load_sweep_results(sweep_summary_path: str) -> pd.DataFrame:
    """Load sweep results into DataFrame"""
    
    with open(sweep_summary_path, 'r') as f:
        sweep_data = json.load(f)
    
    results = []
    
    for result in sweep_data['results']:
        if result['status'] != 'success':
            continue
        
        config = result['config']
        
        # Load training stats from zip
        try:
            zip_path = result['result']
            with zipfile.ZipFile(zip_path, 'r') as zf:
                with zf.open('training_stats.json') as f:
                    stats = json.load(f)
                
                with zf.open('experiment_metadata.json') as f:
                    metadata = json.load(f)
        except:
            continue
        
        # Extract key metrics
        row = {}
        
        # Configuration parameters
        row.update(flatten_config(config))
        
        # Performance metrics
        if 'avg_episode_reward' in stats and stats['avg_episode_reward']:
            row['final_reward'] = stats['avg_episode_reward'][-1]
            row['avg_reward'] = np.mean(stats['avg_episode_reward'])
            row['reward_std'] = np.std(stats['avg_episode_reward'])
        
        if 'avg_vae_loss' in stats and stats['avg_vae_loss']:
            row['final_vae_loss'] = stats['avg_vae_loss'][-1]
            row['avg_vae_loss'] = np.mean(stats['avg_vae_loss'])
        
        if 'avg_kl_loss' in stats and stats['avg_kl_loss']:
            row['final_kl_loss'] = stats['avg_kl_loss'][-1]
            row['avg_kl_loss'] = np.mean(stats['avg_kl_loss'])
        
        # Training efficiency
        row['total_iterations'] = len(stats.get('iteration', []))
        row['model_parameters'] = metadata.get('model_parameters', 0)
        
        # Convergence analysis
        if 'avg_episode_reward' in stats and len(stats['avg_episode_reward']) > 50:
            # Last 25% of training
            last_quarter = stats['avg_episode_reward'][-len(stats['avg_episode_reward'])//4:]
            row['final_quarter_reward'] = np.mean(last_quarter)
            row['final_quarter_std'] = np.std(last_quarter)
        
        results.append(row)
    
    return pd.DataFrame(results)

def flatten_config(config: Dict, prefix: str = '') -> Dict:
    """Flatten nested config dictionary"""
    items = []
    
    for key, value in config.items():
        new_key = f"{prefix}_{key}" if prefix else key
        
        if isinstance(value, dict):
            items.extend(flatten_config(value, new_key).items())
        else:
            items.append((new_key, value))
    
    return dict(items)

def analyze_parameter_importance(df: pd.DataFrame, target_metric: str = 'final_reward') -> pd.DataFrame:
    """Analyze which parameters most affect performance"""
    
    # Find parameter columns (exclude metrics)
    metric_cols = ['final_reward', 'avg_reward', 'reward_std', 'final_vae_loss', 
                   'avg_vae_loss', 'final_kl_loss', 'avg_kl_loss', 'total_iterations',
                   'model_parameters', 'final_quarter_reward', 'final_quarter_std']
    
    param_cols = [col for col in df.columns if col not in metric_cols]
    
    importance = []
    
    for param in param_cols:
        if df[param].dtype in ['object', 'bool'] or df[param].nunique() < 10:
            # Categorical parameter
            groups = df.groupby(param)[target_metric].agg(['mean', 'std', 'count'])
            if len(groups) > 1:
                # Use coefficient of variation between groups
                group_means = groups['mean'].values
                importance_score = np.std(group_means) / np.mean(group_means) if np.mean(group_means) != 0 else 0
                importance.append({
                    'parameter': param,
                    'importance': importance_score,
                    'type': 'categorical',
                    'unique_values': df[param].nunique(),
                    'best_value': groups['mean'].idxmax()
                })
        else:
            # Continuous parameter
            corr = df[param].corr(df[target_metric])
            if not np.isnan(corr):
                importance.append({
                    'parameter': param,
                    'importance': abs(corr),
                    'type': 'continuous',
                    'correlation': corr,
                    'range': f"{df[param].min():.3g} - {df[param].max():.3g}"
                })
    
    importance_df = pd.DataFrame(importance).sort_values('importance', ascending=False)
    return importance_df

def find_best_configurations(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Find best performing configurations"""
    
    if 'final_reward' not in df.columns:
        print("No final_reward column found")
        return df.head(top_n)
    
    # Sort by final reward and get top configurations
    best_configs = df.nlargest(top_n, 'final_reward')
    
    return best_configs

def create_performance_comparison(df: pd.DataFrame, save_path: str = None):
    """Create performance comparison plots"""
    
    if len(df) < 2:
        print("Not enough data for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Parameter importance heatmap
    if 'model_latent_dim' in df.columns and 'model_encoder_hidden' in df.columns:
        pivot = df.pivot_table(values='final_reward', 
                              index='model_latent_dim', 
                              columns='model_encoder_hidden', 
                              aggfunc='mean')
        sns.heatmap(pivot, annot=True, fmt='.3f', ax=axes[0,0])
        axes[0,0].set_title('Final Reward by Latent Dim vs Encoder Hidden')
    
    # 2. Training dynamics
    if 'training_vae_updates' in df.columns:
        df.boxplot(column='final_reward', by='training_vae_updates', ax=axes[0,1])
        axes[0,1].set_title('Final Reward by VAE Updates')
    
    # 3. Model size vs performance
    if 'model_parameters' in df.columns:
        axes[1,0].scatter(df['model_parameters'], df['final_reward'], alpha=0.6)
        axes[1,0].set_xlabel('Model Parameters')
        axes[1,0].set_ylabel('Final Reward')
        axes[1,0].set_title('Model Size vs Performance')
    
    # 4. Learning curves comparison (top 3 vs bottom 3)
    axes[1,1].hist(df['final_reward'], bins=20, alpha=0.7)
    axes[1,1].axvline(df['final_reward'].mean(), color='red', linestyle='--', label='Mean')
    axes[1,1].set_xlabel('Final Reward')
    axes[1,1].set_ylabel('Count')
    axes[1,1].set_title('Final Reward Distribution')
    axes[1,1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plots saved to {save_path}")
    else:
        plt.show()

def generate_analysis_report(sweep_summary_path: str, output_dir: str = "analysis"):
    """Generate complete analysis report"""
    
    print(f"🔍 Analyzing sweep results: {sweep_summary_path}")
    
    # Load data
    df = load_sweep_results(sweep_summary_path)
    
    if len(df) == 0:
        print("❌ No successful experiments found")
        return
    
    print(f"✓ Loaded {len(df)} successful experiments")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 1. Parameter importance analysis
    print("📊 Analyzing parameter importance...")
    importance = analyze_parameter_importance(df, 'final_reward')
    importance_file = output_path / "parameter_importance.csv"
    importance.to_csv(importance_file, index=False)
    
    print(f"Top 5 most important parameters:")
    for _, row in importance.head(5).iterrows():
        print(f"  {row['parameter']}: {row['importance']:.3f}")
    
    # 2. Best configurations
    print("🏆 Finding best configurations...")
    best_configs = find_best_configurations(df, top_n=5)
    best_configs_file = output_path / "best_configurations.csv"
    best_configs.to_csv(best_configs_file, index=False)
    
    print(f"Best configuration achieved reward: {best_configs.iloc[0]['final_reward']:.4f}")
    
    # 3. Performance comparison plots
    print("📈 Creating performance plots...")
    plots_file = output_path / "performance_comparison.png"
    try:
        create_performance_comparison(df, str(plots_file))
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    # 4. Summary statistics
    print("📋 Generating summary...")
    summary = {
        'total_experiments': len(df),
        'best_final_reward': df['final_reward'].max(),
        'worst_final_reward': df['final_reward'].min(),
        'mean_final_reward': df['final_reward'].mean(),
        'std_final_reward': df['final_reward'].std(),
        'most_important_parameter': importance.iloc[0]['parameter'] if len(importance) > 0 else 'unknown'
    }
    
    summary_file = output_path / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📁 Analysis complete! Results saved to: {output_path}")
    print(f"   • Parameter importance: {importance_file}")
    print(f"   • Best configurations: {best_configs_file}")
    print(f"   • Performance plots: {plots_file}")
    print(f"   • Summary: {summary_file}")
    
    return {
        'summary': summary,
        'importance': importance,
        'best_configs': best_configs,
        'all_data': df
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze VariBAD parameter sweep results")
    parser.add_argument('sweep_file', help='Path to sweep summary JSON file')
    parser.add_argument('--output', '-o', default='analysis', help='Output directory')
    
    args = parser.parse_args()
    
    if not Path(args.sweep_file).exists():
        print(f"❌ Sweep file not found: {args.sweep_file}")
        return
    
    generate_analysis_report(args.sweep_file, args.output)

if __name__ == "__main__":
    main()