"""
Diagnostic script to analyze portfolio weight distributions from backtest.
Checks if normalization worked correctly and visualizes weight behavior.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_backtest_data(csv_path):
    """Load backtest CSV and extract weight columns."""
    df = pd.read_csv(csv_path)
    
    # Find all weight columns (weight_0, weight_1, ..., weight_N)
    weight_cols = [col for col in df.columns if col.startswith('weight_')]
    num_assets = len(weight_cols)
    
    print(f"Found {num_assets} assets in backtest")
    print(f"Total timesteps: {len(df)}")
    
    return df, weight_cols

def check_normalization(df, weight_cols):
    """Verify that weights satisfy budget constraint."""
    weights = df[weight_cols].values
    
    # Check: sum(|w_i|) + w_cash = 1
    abs_sum = np.abs(weights).sum(axis=1)
    cash = 1.0 - abs_sum
    
    print("\n=== NORMALIZATION CHECK ===")
    print(f"sum(|weights|) statistics:")
    print(f"  Mean: {abs_sum.mean():.6f}")
    print(f"  Std:  {abs_sum.std():.6f}")
    print(f"  Min:  {abs_sum.min():.6f}")
    print(f"  Max:  {abs_sum.max():.6f}")
    print(f"\nImplied cash position:")
    print(f"  Mean: {cash.mean():.6f}")
    print(f"  Std:  {cash.std():.6f}")
    print(f"  Min:  {cash.min():.6f}")
    print(f"  Max:  {cash.max():.6f}")
    
    # Check if normalization is working (should be close to 1.0)
    normalization_error = np.abs(abs_sum + cash - 1.0)
    print(f"\nNormalization error |sum(|w|) + cash - 1|:")
    print(f"  Mean: {normalization_error.mean():.9f}")
    print(f"  Max:  {normalization_error.max():.9f}")
    
    if normalization_error.max() < 1e-6:
        print("✅ Normalization working correctly!")
    else:
        print("⚠️  Normalization may have issues")
    
    return weights, cash

def analyze_weight_variation(weights):
    """Analyze how much weights change over time."""
    print("\n=== WEIGHT VARIATION ANALYSIS ===")
    
    # Per-asset statistics
    print("\nPer-asset weight statistics:")
    for i in range(min(5, weights.shape[1])):  # Show first 5 assets
        w = weights[:, i]
        print(f"  Asset {i}: mean={w.mean():+.6f}, std={w.std():.6f}, "
              f"range=[{w.min():+.6f}, {w.max():+.6f}]")
    
    # Turnover (step-to-step changes)
    turnover = np.abs(np.diff(weights, axis=0)).sum(axis=1)
    print(f"\nTurnover per step (sum of |Δw_i|):")
    print(f"  Mean: {turnover.mean():.6f}")
    print(f"  Std:  {turnover.std():.6f}")
    print(f"  Min:  {turnover.min():.6f}")
    print(f"  Max:  {turnover.max():.6f}")
    
    # Check if weights are "frozen"
    max_change = np.abs(np.diff(weights, axis=0)).max()
    print(f"\nMaximum single weight change: {max_change:.9f}")
    
    if max_change < 1e-3:
        print("⚠️  WARNING: Weights changing by <0.001 - policy may be frozen!")
    elif max_change < 0.01:
        print("⚠️  Weights changing slowly (< 1%)")
    else:
        print("✅ Weights showing healthy variation")
    
    return turnover

def analyze_exposure(weights):
    """Analyze long/short exposure patterns."""
    print("\n=== EXPOSURE ANALYSIS ===")
    
    long_exposure = (weights * (weights > 0)).sum(axis=1)
    short_exposure = np.abs(weights * (weights < 0)).sum(axis=1)
    net_exposure = weights.sum(axis=1)
    gross_exposure = np.abs(weights).sum(axis=1)
    
    print(f"Long exposure:")
    print(f"  Mean: {long_exposure.mean():.4f}, Std: {long_exposure.std():.4f}")
    print(f"Short exposure:")
    print(f"  Mean: {short_exposure.mean():.4f}, Std: {short_exposure.std():.4f}")
    print(f"Net exposure:")
    print(f"  Mean: {net_exposure.mean():+.4f}, Std: {net_exposure.std():.4f}")
    print(f"Gross exposure:")
    print(f"  Mean: {gross_exposure.mean():.4f}, Std: {gross_exposure.std():.4f}")

def plot_weight_distributions(df, weight_cols, output_dir):
    """Create diagnostic plots for weight distributions."""
    weights = df[weight_cols].values
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Weight distribution at different time points
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Weight Distributions Over Time', fontsize=16)
    
    time_points = [0, len(df)//5, 2*len(df)//5, 3*len(df)//5, 4*len(df)//5, len(df)-1]
    labels = ['Start', '20%', '40%', '60%', '80%', 'End']
    
    for idx, (t, label) in enumerate(zip(time_points, labels)):
        ax = axes[idx // 3, idx % 3]
        w = weights[t]
        ax.hist(w, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('Weight')
        ax.set_ylabel('Count')
        ax.set_title(f'{label} (t={t})')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        ax.text(0.02, 0.98, f'μ={w.mean():.4f}\nσ={w.std():.4f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'weight_distributions_over_time.png', dpi=150)
    print(f"✅ Saved: {output_dir / 'weight_distributions_over_time.png'}")
    plt.close()
    
    # 2. Weight trajectories (first 10 assets)
    fig, ax = plt.subplots(figsize=(15, 6))
    for i in range(min(10, weights.shape[1])):
        ax.plot(weights[:, i], alpha=0.6, label=f'Asset {i}')
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Weight')
    ax.set_title('Portfolio Weight Trajectories (First 10 Assets)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'weight_trajectories.png', dpi=150)
    print(f"✅ Saved: {output_dir / 'weight_trajectories.png'}")
    plt.close()
    
    # 3. Turnover over time
    turnover = np.abs(np.diff(weights, axis=0)).sum(axis=1)
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(turnover, alpha=0.7)
    ax.axhline(turnover.mean(), color='red', linestyle='--', 
               label=f'Mean = {turnover.mean():.4f}')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Turnover (sum of |Δw|)')
    ax.set_title('Portfolio Turnover Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'turnover_over_time.png', dpi=150)
    print(f"✅ Saved: {output_dir / 'turnover_over_time.png'}")
    plt.close()
    
    # 4. Heatmap of weights over time (subsampled)
    subsample = max(1, len(df) // 100)  # Show ~100 timesteps
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.heatmap(weights[::subsample].T, cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Weight'}, ax=ax)
    ax.set_xlabel('Timestep (subsampled)')
    ax.set_ylabel('Asset')
    ax.set_title('Portfolio Weights Heatmap (Subsampled)')
    plt.tight_layout()
    plt.savefig(output_dir / 'weights_heatmap.png', dpi=150)
    print(f"✅ Saved: {output_dir / 'weights_heatmap.png'}")
    plt.close()
    
    # 5. Weight change distribution
    weight_changes = np.diff(weights, axis=0).flatten()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(weight_changes, bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Weight Change (Δw)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Step-to-Step Weight Changes')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'Mean: {weight_changes.mean():.6f}\n'
    stats_text += f'Std: {weight_changes.std():.6f}\n'
    stats_text += f'Max: {weight_changes.max():.6f}\n'
    stats_text += f'Min: {weight_changes.min():.6f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'weight_changes_distribution.png', dpi=150)
    print(f"✅ Saved: {output_dir / 'weight_changes_distribution.png'}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze backtest weight distributions')
    parser.add_argument('csv_path', type=str, help='Path to backtest CSV file')
    parser.add_argument('--output_dir', type=str, default='weight_diagnostics',
                       help='Output directory for plots')
    args = parser.parse_args()
    
    print("="*80)
    print("PORTFOLIO WEIGHT DIAGNOSTIC TOOL")
    print("="*80)
    
    # Load data
    df, weight_cols = load_backtest_data(args.csv_path)
    
    # Check normalization
    weights, cash = check_normalization(df, weight_cols)
    
    # Analyze variation
    turnover = analyze_weight_variation(weights)
    
    # Analyze exposure
    analyze_exposure(weights)
    
    # Create plots
    print("\n=== GENERATING PLOTS ===")
    plot_weight_distributions(df, weight_cols, args.output_dir)
    
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE!")
    print("="*80)
    
    # Final summary
    max_change = np.abs(np.diff(weights, axis=0)).max()
    if max_change < 1e-3:
        print("\n⚠️  CRITICAL: Weights are essentially frozen!")
        print("   Likely causes: double tanh, collapsed latent, or zero entropy")
        print("   → Remove first tanh from policy network")
        print("   → Check latent.std() > 0.1")
        print("   → Increase entropy coefficient")
    elif max_change < 0.01:
        print("\n⚠️  WARNING: Weights changing very slowly")
        print("   Policy may be too conservative")
        print("   → Check transaction costs")
        print("   → Increase exploration during training")
    else:
        print("\n✅ Weights showing reasonable variation")
        print(f"   Max change: {max_change:.4f}")

if __name__ == "__main__":
    main()