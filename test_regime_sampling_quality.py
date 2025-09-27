# task_diversity_check.py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add your project to path
sys.path.append('.')

from environments.data import PortfolioDataset
from config import TrainingConfig, experiment_to_training_config, ExperimentConfig

def analyze_task_diversity(asset_class='sp500', n_tasks=100, seq_len=200):
    """
    Sample tasks and analyze their statistical properties to determine
    if they constitute meaningfully different 'tasks' for meta-learning.
    """
    print(f"\n{'='*60}")
    print(f"Task Diversity Analysis: {asset_class}")
    print(f"{'='*60}\n")
    
    # Load dataset
    if asset_class == "crypto":
        dataset = PortfolioDataset(
            asset_class=asset_class,
            data_path=f"environments/data/{asset_class}_rl_ready_cleaned.parquet",
            proportional=True,
            proportions=(0.7, 0.2, 0.1)
        )
    else:
        dataset = PortfolioDataset(
            asset_class=asset_class,
            data_path=f"environments/data/{asset_class}_rl_ready_cleaned.parquet",
            train_end='2015-12-31',
            val_end='2020-12-31'
        )
    
    # Get training split
    train_split = dataset.get_split('train')
    
    print(f"Dataset info:")
    print(f"  Total days: {len(train_split)}")
    print(f"  Assets: {train_split.num_assets}")
    print(f"  Features: {train_split.num_features}")
    print(f"  Sequence length: {seq_len}\n")
    
    # Sample tasks
    task_stats = []
    max_start = len(train_split) - seq_len
    
    if max_start <= 0:
        print(f"ERROR: Dataset too short for seq_len={seq_len}")
        return
    
    print(f"Sampling {n_tasks} tasks...\n")
    
    for i in range(n_tasks):
        # Sample random window (mimics MetaEnv.sample_task)
        start = np.random.randint(0, max_start)
        end = start + seq_len
        
        # Get window data
        window = train_split.get_window(start, end)
        prices = window['raw_prices']  # Shape: (seq_len, num_assets)
        
        # Compute returns
        returns = (prices[1:] / prices[:-1]) - 1  # Shape: (seq_len-1, num_assets)
        
        # Aggregate across assets (portfolio-level statistics)
        portfolio_returns = returns.mean(axis=1)  # Mean return across assets per timestep
        
        # Task-level statistics
        task_stats.append({
            'task_id': i,
            'start_idx': start,
            'mean_return': portfolio_returns.mean(),
            'volatility': portfolio_returns.std(),
            'max_return': portfolio_returns.max(),
            'min_return': portfolio_returns.min(),
            'max_drawdown': compute_max_drawdown(prices.mean(axis=1)),  # Portfolio-level drawdown
            'positive_days_pct': (portfolio_returns > 0).mean(),
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(task_stats)
    
    # Analysis
    print("Task Statistics Summary:")
    print("="*60)
    print(df.describe().to_string())
    print("\n")
    
    # Diversity metrics
    print("Diversity Analysis:")
    print("="*60)
    
    # Coefficient of variation (std/mean) - higher = more diverse
    for col in ['mean_return', 'volatility', 'max_drawdown']:
        cv = df[col].std() / (abs(df[col].mean()) + 1e-8)
        print(f"{col:20s} - CV: {cv:.3f}")
    
    # Check for clustering (simple heuristic)
    vol_range = df['volatility'].max() - df['volatility'].min()
    ret_range = df['mean_return'].max() - df['mean_return'].min()
    
    print(f"\nRange Analysis:")
    print(f"  Volatility range: {vol_range:.6f} ({vol_range/df['volatility'].mean()*100:.1f}% of mean)")
    print(f"  Return range: {ret_range:.6f} ({abs(ret_range)/abs(df['mean_return'].mean())*100:.1f}% of mean)")
    
    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("="*60)
    
    # Volatility check
    vol_cv = df['volatility'].std() / df['volatility'].mean()
    if vol_cv > 0.3:
        print("✓ HIGH task diversity in volatility (CV > 0.3)")
        print("  → Tasks have significantly different risk profiles")
    elif vol_cv > 0.15:
        print("~ MODERATE task diversity in volatility (0.15 < CV < 0.3)")
        print("  → Some variation but not strongly differentiated")
    else:
        print("✗ LOW task diversity in volatility (CV < 0.15)")
        print("  → Tasks are very similar in risk characteristics")
    
    # Return check
    ret_std = df['mean_return'].std()
    ret_mean_abs = abs(df['mean_return'].mean())
    if ret_std > ret_mean_abs:
        print("✓ HIGH task diversity in returns")
        print("  → Tasks span different return regimes")
    else:
        print("✗ LOW task diversity in returns")
        print("  → Tasks have similar return characteristics")
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("="*60)
    
    if vol_cv > 0.3 or ret_std > ret_mean_abs:
        print("✓ Your tasks have sufficient diversity for VariBAD")
        print("  → Proceed with VAE asymmetric encoding fix")
    else:
        print("✗ Your tasks lack diversity for meta-learning")
        print("  → Consider regime-based task sampling before VAE fix")
        print("  → Or accept you're doing recurrent RL, not VariBAD")
    
    # Visualize distribution (optional)
    print("\nTask Distribution (volatility vs returns):")
    plot_task_distribution(df)
    
    return df

def compute_max_drawdown(prices):
    """Compute maximum drawdown from price series."""
    peak = np.maximum.accumulate(prices)
    drawdown = (peak - prices) / peak
    return drawdown.max()

def plot_task_distribution(df):
    """Simple text-based scatter plot of task characteristics."""
    # Normalize to 0-50 range for plotting
    vol_norm = ((df['volatility'] - df['volatility'].min()) / 
                (df['volatility'].max() - df['volatility'].min()) * 50).astype(int)
    ret_norm = ((df['mean_return'] - df['mean_return'].min()) / 
                (df['mean_return'].max() - df['mean_return'].min()) * 20).astype(int)
    
    # Create grid
    grid = [[' ' for _ in range(52)] for _ in range(22)]
    
    # Plot points
    for v, r in zip(vol_norm, ret_norm):
        if 0 <= v < 51 and 0 <= r < 21:
            grid[20-r][v] = '*'
    
    # Print
    print("  " + "-"*52)
    for i, row in enumerate(grid):
        if i == 10:
            print("R |" + "".join(row) + "|")
        else:
            print("  |" + "".join(row) + "|")
    print("  " + "-"*52)
    print("   " + " "*20 + "Volatility →")
    print("\n  * = task, R = returns axis")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset_class', type=str, default='sp500', 
                       choices=['sp500', 'crypto'])
    parser.add_argument('--n_tasks', type=int, default=100,
                       help='Number of tasks to sample')
    parser.add_argument('--seq_len', type=int, default=200,
                       help='Task sequence length')
    
    args = parser.parse_args()
    
    df = analyze_task_diversity(
        asset_class=args.asset_class,
        n_tasks=args.n_tasks,
        seq_len=args.seq_len
    )
    
    # Save results
    output_file = f"task_diversity_{args.asset_class}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")