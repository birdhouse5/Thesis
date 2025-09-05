#!/usr/bin/env python3
"""
Analyze Phase 1 Optuna results to inform Phase 2 parameter selection.
Usage: python analyze_phase1_results.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_phase1_results():
    """Load the Optuna Phase 1 results"""
    results_dir = Path("optuna_results")
    
    # Find the most recent results file
    csv_files = list(results_dir.glob("*_all_trials.csv"))
    if not csv_files:
        raise FileNotFoundError("No Optuna results CSV found in optuna_results/")
    
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 Loading results from: {latest_csv}")
    
    df = pd.read_csv(latest_csv)
    
    # Filter only completed trials
    completed = df[df['state'] == 'COMPLETE'].copy()
    print(f"✅ Completed trials: {len(completed)}/{len(df)} ({len(completed)/len(df)*100:.1f}%)")
    
    return completed

def analyze_architecture_patterns(df):
    """Analyze which architecture combinations work best"""
    print("\n" + "="*60)
    print("🏗️  ARCHITECTURE ANALYSIS")
    print("="*60)
    
    # Top 10 performing trials
    top_10 = df.nlargest(10, 'value')
    print("\n🏆 Top 10 Trials:")
    print(top_10[['trial_number', 'value', 'param_latent_dim', 'param_hidden_dim', 
                  'param_vae_lr', 'param_policy_lr']].round(6))
    
    # Architecture combination analysis
    print("\n📊 Architecture Performance Summary:")
    arch_stats = df.groupby(['param_latent_dim', 'param_hidden_dim']).agg({
        'value': ['count', 'mean', 'std', 'max', 'min']
    }).round(4)
    arch_stats.columns = ['count', 'mean_sharpe', 'std_sharpe', 'max_sharpe', 'min_sharpe']
    arch_stats = arch_stats.sort_values('mean_sharpe', ascending=False)
    print(arch_stats)
    
    # Learning rate analysis for best architectures
    print("\n📈 Learning Rate Analysis for Top Architectures:")
    top_archs = arch_stats.head(3).index
    for latent_dim, hidden_dim in top_archs:
        subset = df[(df['param_latent_dim'] == latent_dim) & 
                   (df['param_hidden_dim'] == hidden_dim)]
        if len(subset) > 0:
            print(f"\n  Architecture: latent={latent_dim}, hidden={hidden_dim}")
            print(f"    VAE LR range: {subset['param_vae_lr'].min():.2e} - {subset['param_vae_lr'].max():.2e}")
            print(f"    Policy LR range: {subset['param_policy_lr'].min():.2e} - {subset['param_policy_lr'].max():.2e}")
            print(f"    Best trial value: {subset['value'].max():.4f}")
    
    return top_archs

def identify_phase2_parameters(df, top_archs):
    """Determine Phase 2 parameter ranges based on Phase 1 results"""
    print("\n" + "="*60)
    print("🎯 PHASE 2 PARAMETER RECOMMENDATIONS")
    print("="*60)
    
    # Get best trial details
    best_trial = df.loc[df['value'].idxmax()]
    print(f"\n🥇 Best Trial #{int(best_trial['trial_number'])} (Sharpe: {best_trial['value']:.4f}):")
    print(f"    latent_dim: {int(best_trial['param_latent_dim'])}")
    print(f"    hidden_dim: {int(best_trial['param_hidden_dim'])}")
    print(f"    vae_lr: {best_trial['param_vae_lr']:.6f}")
    print(f"    policy_lr: {best_trial['param_policy_lr']:.6f}")
    
    # Analyze top 20% of trials
    top_20_pct = df.nlargest(max(15, len(df)//5), 'value')
    
    print(f"\n📈 Analysis of Top {len(top_20_pct)} Trials:")
    
    # Architecture preferences
    arch_counts = top_20_pct.groupby(['param_latent_dim', 'param_hidden_dim']).size()
    print(f"\n  Most Successful Architectures:")
    for (latent, hidden), count in arch_counts.sort_values(ascending=False).head(5).items():
        pct = count / len(top_20_pct) * 100
        print(f"    latent={latent}, hidden={hidden}: {count} trials ({pct:.1f}%)")
    
    # Learning rate ranges from top performers
    vae_lr_stats = top_20_pct['param_vae_lr'].describe()
    policy_lr_stats = top_20_pct['param_policy_lr'].describe()
    
    print(f"\n  Learning Rate Ranges (Top {len(top_20_pct)} trials):")
    print(f"    VAE LR: {vae_lr_stats['min']:.2e} - {vae_lr_stats['max']:.2e} (median: {vae_lr_stats['50%']:.2e})")
    print(f"    Policy LR: {policy_lr_stats['min']:.2e} - {policy_lr_stats['max']:.2e} (median: {policy_lr_stats['50%']:.2e})")
    
    # Generate Phase 2 recommendations
    print(f"\n🎯 RECOMMENDED PHASE 2 PARAMETERS:")
    
    # Fixed architecture (best performing)
    best_latent = int(best_trial['param_latent_dim'])
    best_hidden = int(best_trial['param_hidden_dim'])
    
    # Learning rate ranges (expand around successful region)
    vae_lr_center = vae_lr_stats['50%']
    policy_lr_center = policy_lr_stats['50%']
    
    print(f"""
    # Fixed from Phase 1 (best architecture):
    latent_dim = {best_latent}
    hidden_dim = {best_hidden}
    
    # Narrow LR search around successful region:
    vae_lr: [{vae_lr_center/3:.2e}, {vae_lr_center*3:.2e}]
    policy_lr: [{policy_lr_center/3:.2e}, {policy_lr_center*3:.2e}]
    
    # New parameters to optimize:
    vae_beta: [0.001, 0.01, 0.1, 0.5, 1.0]
    seq_len: [30, 60, 90, 120] 
    batch_size: [1024, 2048, 4096, 8192]
    episodes_per_task: [3, 5, 8, 12]
    ppo_epochs: [2, 4, 8]
    """)
    
    return {
        'best_latent_dim': best_latent,
        'best_hidden_dim': best_hidden,
        'vae_lr_range': [vae_lr_center/3, vae_lr_center*3],
        'policy_lr_range': [policy_lr_center/3, policy_lr_center*3],
        'vae_lr_center': vae_lr_center,
        'policy_lr_center': policy_lr_center
    }

def generate_phase2_config_template(recommendations):
    """Generate Phase 2 Optuna config template"""
    
    config_template = f'''
# Phase 2 Optuna Configuration
# Based on Phase 1 results - optimize VAE and training dynamics

class OptunaPhase2Config:
    def __init__(self, trial: optuna.Trial):
        # FIXED from Phase 1 (best architecture)
        self.latent_dim = {recommendations['best_latent_dim']}
        self.hidden_dim = {recommendations['best_hidden_dim']}
        
        # NARROWED learning rates (around successful region)
        self.vae_lr = trial.suggest_float('vae_lr', {recommendations['vae_lr_range'][0]:.2e}, {recommendations['vae_lr_range'][1]:.2e}, log=True)
        self.policy_lr = trial.suggest_float('policy_lr', {recommendations['policy_lr_range'][0]:.2e}, {recommendations['policy_lr_range'][1]:.2e}, log=True)
        
        # NEW parameters to optimize
        self.vae_beta = trial.suggest_float('vae_beta', 0.001, 1.0, log=True)
        self.seq_len = trial.suggest_categorical('seq_len', [30, 60, 90, 120])
        self.batch_size = trial.suggest_categorical('batch_size', [1024, 2048, 4096, 8192])
        self.vae_batch_size = trial.suggest_categorical('vae_batch_size', [256, 512, 1024, 2048])
        self.episodes_per_task = trial.suggest_categorical('episodes_per_task', [3, 5, 8, 12])
        self.ppo_epochs = trial.suggest_categorical('ppo_epochs', [2, 4, 8])
        self.vae_update_freq = trial.suggest_categorical('vae_update_freq', [1, 2, 5])
        
        # Ensure vae_batch_size <= batch_size
        self.vae_batch_size = min(self.vae_batch_size, self.batch_size // 2)
        
        # Adjust horizons based on seq_len
        self.max_horizon = min(self.seq_len - 10, int(self.seq_len * 0.8))
        self.min_horizon = max(self.max_horizon - 15, self.max_horizon // 2)
        
        # Fixed base parameters (keep successful settings)
        self.data_path = "environments/data/sp500_rl_ready_cleaned.parquet"
        self.train_end = '2015-12-31'
        self.val_end = '2020-12-31'
        self.num_assets = 30
        self.device = "cuda"
        
        # Longer training for Phase 2 (more episodes to see effects)
        self.max_episodes = 3000  # Increased from 2000
        self.val_interval = 500
        self.val_episodes = 50
        
        # Keep successful PPO settings
        self.ppo_clip_ratio = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.gae_lambda = 0.95
        self.discount_factor = 0.99
        
        # Experiment naming
        self.exp_name = f"phase2_t{{trial.number}}_beta{{self.vae_beta:.3f}}_seq{{self.seq_len}}"
'''
    
    return config_template

def main():
    """Analyze Phase 1 and generate Phase 2 recommendations"""
    try:
        # Load and analyze results
        df = load_phase1_results()
        top_archs = analyze_architecture_patterns(df)
        recommendations = identify_phase2_parameters(df, top_archs)
        
        # Generate Phase 2 template
        phase2_template = generate_phase2_config_template(recommendations)
        
        # Save Phase 2 config template
        output_file = Path("optuna_phase2_config.py")
        with open(output_file, 'w') as f:
            f.write("import optuna\n")
            f.write(phase2_template)
        
        print(f"\n💾 Phase 2 config template saved to: {output_file}")
        
        # Summary recommendations
        print(f"\n" + "="*60)
        print("📋 NEXT STEPS FOR PHASE 2")
        print("="*60)
        print(f"""
1. 🎯 Fix architecture to winning combination:
   - latent_dim = {recommendations['best_latent_dim']}
   - hidden_dim = {recommendations['best_hidden_dim']}

2. 🔧 Focus on VAE and training dynamics:
   - vae_beta: Critical parameter that was fixed at 0.1 in Phase 1
   - seq_len: Context length affects task adaptation
   - batch_size: Memory vs. stability trade-offs
   - episodes_per_task: Exploration vs. exploitation

3. ⚡ Suggested Phase 2 setup:
   - Trials: 150-200 (more complex parameter space)
   - Max episodes: 3000 (longer to see VAE effects)
   - Parallel jobs: 10-12 (slightly fewer due to longer trials)

4. 🚀 Run Phase 2:
   export TOTAL_TRIALS=150
   export N_JOBS=10
   python optuna_phase2.py
""")
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        print("Please ensure optuna_results/ directory contains the trial CSV files")

if __name__ == "__main__":
    main()