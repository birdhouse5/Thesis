# analyze_splits.py - Utility to analyze dataset splits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def analyze_dataset_splits(data_path, train_end='2015-12-31', val_end='2020-12-31'):
    """
    Analyze the temporal distribution and characteristics of dataset splits.
    
    Args:
        data_path: Path to the dataset parquet file
        train_end: End date for training split
        val_end: End date for validation split
    """
    print("🔍 Analyzing Dataset Splits")
    print("=" * 50)
    
    # Load data
    df = pd.read_parquet(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"📊 Full Dataset:")
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Tickers: {df['ticker'].nunique()}")
    print(f"   Days: {df['date'].nunique()}")
    
    # Define split boundaries
    train_end_date = pd.to_datetime(train_end)
    val_end_date = pd.to_datetime(val_end)
    
    # Create splits
    train_data = df[df['date'] <= train_end_date]
    val_data = df[(df['date'] > train_end_date) & (df['date'] <= val_end_date)]
    test_data = df[df['date'] > val_end_date]
    
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    print(f"\n📅 Split Configuration:")
    print(f"   Train: up to {train_end}")
    print(f"   Val: {train_end} to {val_end}")
    print(f"   Test: after {val_end}")
    
    # Analyze each split
    print(f"\n📈 Split Analysis:")
    total_rows = len(df)
    
    for split_name, split_data in splits.items():
        if len(split_data) == 0:
            print(f"   ⚠️  {split_name.upper()}: NO DATA")
            continue
            
        pct = len(split_data) / total_rows * 100
        date_range = f"{split_data['date'].min().date()} to {split_data['date'].max().date()}"
        days = split_data['date'].nunique()
        
        print(f"   {split_name.upper()}:")
        print(f"      Rows: {len(split_data):,} ({pct:.1f}%)")
        print(f"      Date range: {date_range}")
        print(f"      Days: {days:,}")
        print(f"      Tickers: {split_data['ticker'].nunique()}")
        
        # Check for data quality issues
        missing_data = split_data.isnull().sum().sum()
        if missing_data > 0:
            print(f"      ⚠️  Missing values: {missing_data}")
        
        # Check rectangularity
        expected_rows = days * split_data['ticker'].nunique()
        if len(split_data) != expected_rows:
            print(f"      ⚠️  Not rectangular: {len(split_data)} vs {expected_rows} expected")
    
    # Market statistics by split
    print(f"\n📊 Market Statistics by Split:")
    
    for split_name, split_data in splits.items():
        if len(split_data) == 0:
            continue
            
        # Calculate market statistics
        daily_returns = split_data.groupby('date')['returns'].mean()
        market_vol = daily_returns.std() * np.sqrt(252)  # Annualized
        
        print(f"   {split_name.upper()}:")
        print(f"      Avg daily return: {daily_returns.mean():.4f}")
        print(f"      Volatility (ann.): {market_vol:.4f}")
        print(f"      Min daily return: {daily_returns.min():.4f}")
        print(f"      Max daily return: {daily_returns.max():.4f}")
    
    # Check for temporal gaps
    print(f"\n🕐 Temporal Continuity Check:")
    
    all_dates = sorted(df['date'].unique())
    gaps = []
    
    for i in range(1, len(all_dates)):
        gap_days = (all_dates[i] - all_dates[i-1]).days
        if gap_days > 7:  # More than a week gap (accounting for weekends)
            gaps.append((all_dates[i-1], all_dates[i], gap_days))
    
    if gaps:
        print(f"   Found {len(gaps)} temporal gaps > 7 days:")
        for start_date, end_date, gap_days in gaps[:5]:  # Show first 5
            print(f"      {start_date.date()} to {end_date.date()} ({gap_days} days)")
        if len(gaps) > 5:
            print(f"      ... and {len(gaps) - 5} more")
    else:
        print(f"   ✅ No significant temporal gaps found")
    
    # Feature availability across splits
    print(f"\n🎯 Feature Analysis:")
    
    normalized_features = [col for col in df.columns if col.endswith('_norm')]
    print(f"   Normalized features: {len(normalized_features)}")
    
    # Check feature completeness across splits
    for split_name, split_data in splits.items():
        if len(split_data) == 0:
            continue
            
        missing_features = []
        for feature in normalized_features:
            if feature in split_data.columns:
                missing_pct = split_data[feature].isnull().sum() / len(split_data) * 100
                if missing_pct > 5:  # More than 5% missing
                    missing_features.append((feature, missing_pct))
        
        if missing_features:
            print(f"   {split_name.upper()} - Features with >5% missing:")
            for feature, pct in missing_features[:3]:  # Show top 3
                print(f"      {feature}: {pct:.1f}% missing")
        else:
            print(f"   {split_name.upper()}: ✅ All features complete")
    
    return splits

def plot_split_timeline(data_path, train_end='2015-12-31', val_end='2020-12-31', save_path=None):
    """Create a timeline visualization of the splits"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        df = pd.read_parquet(data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Get daily market returns
        daily_returns = df.groupby('date')['returns'].mean().reset_index()
        daily_returns = daily_returns.sort_values('date')
        
        # Define split boundaries
        train_end_date = pd.to_datetime(train_end)
        val_end_date = pd.to_datetime(val_end)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Timeline with split boundaries
        ax1.plot(daily_returns['date'], daily_returns['returns'].cumsum(), 
                linewidth=1, color='blue', alpha=0.7)
        ax1.axvline(train_end_date, color='red', linestyle='--', linewidth=2, 
                   label=f'Train end ({train_end})')
        ax1.axvline(val_end_date, color='orange', linestyle='--', linewidth=2, 
                   label=f'Val end ({val_end})')
        
        ax1.fill_between(daily_returns['date'], 
                        daily_returns['returns'].cumsum().min(), 
                        daily_returns['returns'].cumsum().max(),
                        where=(daily_returns['date'] <= train_end_date),
                        alpha=0.2, color='green', label='Train')
        
        ax1.fill_between(daily_returns['date'], 
                        daily_returns['returns'].cumsum().min(), 
                        daily_returns['returns'].cumsum().max(),
                        where=((daily_returns['date'] > train_end_date) & 
                               (daily_returns['date'] <= val_end_date)),
                        alpha=0.2, color='orange', label='Validation')
        
        ax1.fill_between(daily_returns['date'], 
                        daily_returns['returns'].cumsum().min(), 
                        daily_returns['returns'].cumsum().max(),
                        where=(daily_returns['date'] > val_end_date),
                        alpha=0.2, color='red', label='Test')
        
        ax1.set_title('Dataset Timeline with Train-Val-Test Splits')
        ax1.set_ylabel('Cumulative Returns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Rolling volatility
        rolling_vol = daily_returns['returns'].rolling(252).std() * np.sqrt(252)
        ax2.plot(daily_returns['date'], rolling_vol, linewidth=1, color='purple')
        ax2.axvline(train_end_date, color='red', linestyle='--', linewidth=2)
        ax2.axvline(val_end_date, color='orange', linestyle='--', linewidth=2)
        
        ax2.set_title('Rolling 1-Year Volatility')
        ax2.set_ylabel('Annualized Volatility')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Timeline plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("matplotlib not available - skipping timeline plot")
    except Exception as e:
        print(f"Error creating timeline plot: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze VariBAD dataset splits")
    parser.add_argument('--data', type=str, 
                       default="environments/data/sp500_rl_ready_cleaned.parquet",
                       help='Path to dataset parquet file')
    parser.add_argument('--train-end', type=str, default='2015-12-31',
                       help='End date for training split')
    parser.add_argument('--val-end', type=str, default='2020-12-31',
                       help='End date for validation split')
    parser.add_argument('--plot', action='store_true',
                       help='Create timeline visualization')
    parser.add_argument('--save-plot', type=str,
                       help='Save timeline plot to file')
    
    args = parser.parse_args()
    
    if not Path(args.data).exists():
        print(f"❌ Dataset file not found: {args.data}")
        print("Run data preparation first or check the path")
        return
    
    # Analyze splits
    splits = analyze_dataset_splits(args.data, args.train_end, args.val_end)
    
    # Create timeline plot if requested
    if args.plot or args.save_plot:
        plot_split_timeline(args.data, args.train_end, args.val_end, args.save_plot)
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()