# dataset_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from environments.data import PortfolioDataset

def analyze_sp500_dataset(data_path="environments/data/sp500_rl_ready_cleaned.parquet"):
    """Comprehensive statistical analysis of SP500 dataset."""
    
    # Load dataset
    dataset = PortfolioDataset(
        asset_class="sp500",
        data_path=data_path,
        train_end="2015-12-31",
        val_end="2020-12-31"
    )
    
    splits = dataset.get_all_splits()
    full_data = dataset.full_data
    
    # ============ BASIC STATISTICS ============
    print("="*80)
    print("SP500 DATASET STATISTICS")
    print("="*80)
    
    print(f"\n1. DATASET STRUCTURE")
    print(f"   Total rows: {len(full_data):,}")
    print(f"   Date range: {full_data['date'].min().date()} to {full_data['date'].max().date()}")
    print(f"   Trading days: {full_data['date'].nunique():,}")
    print(f"   Number of tickers: {full_data['ticker'].nunique()}")
    print(f"   Features: {len([c for c in full_data.columns if c.endswith('_norm')])} normalized features")
    
    # Split statistics
    print(f"\n2. TEMPORAL SPLITS")
    for split_name in ['train', 'val', 'test']:
        split = splits[split_name]
        info = split.get_split_info()
        print(f"   {split_name.upper()}: {info['num_days']} days, "
              f"{info['date_range'][0]} to {info['date_range'][1]}")
    
    # ============ RETURN STATISTICS ============
    print(f"\n3. RETURN CHARACTERISTICS")
    
    returns_stats = full_data.groupby('ticker')['returns'].agg([
        'count', 'mean', 'std', 'min', 'max',
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75),
        lambda x: stats.skew(x.dropna()),
        lambda x: stats.kurtosis(x.dropna())
    ])
    returns_stats.columns = ['count', 'mean', 'std', 'min', 'max', 'q25', 'q75', 'skew', 'kurtosis']
    
    print(f"\n   Per-Ticker Return Statistics (daily):")
    print(f"   Mean return: {returns_stats['mean'].mean():.6f} ± {returns_stats['mean'].std():.6f}")
    print(f"   Mean volatility: {returns_stats['std'].mean():.4f} ± {returns_stats['std'].std():.4f}")
    print(f"   Skewness: {returns_stats['skew'].mean():.3f} (avg)")
    print(f"   Kurtosis: {returns_stats['kurtosis'].mean():.3f} (avg, excess)")
    
    # Annualized metrics
    print(f"\n   Annualized Metrics (252 trading days):")
    annual_return = returns_stats['mean'].mean() * 252
    annual_vol = returns_stats['std'].mean() * np.sqrt(252)
    sharpe = annual_return / annual_vol
    print(f"   Average annual return: {annual_return:.2%}")
    print(f"   Average annual volatility: {annual_vol:.2%}")
    print(f"   Average Sharpe ratio: {sharpe:.3f}")
    
    # ============ CORRELATION ANALYSIS ============
    print(f"\n4. CORRELATION STRUCTURE")
    
    # Pivot returns for correlation
    returns_pivot = full_data.pivot(index='date', columns='ticker', values='returns')
    corr_matrix = returns_pivot.corr()
    
    # Off-diagonal correlations
    mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
    off_diag_corr = corr_matrix.where(mask).values.flatten()
    off_diag_corr = off_diag_corr[~np.isnan(off_diag_corr)]
    
    print(f"   Pairwise correlation (returns):")
    print(f"   Mean: {off_diag_corr.mean():.3f}")
    print(f"   Median: {np.median(off_diag_corr):.3f}")
    print(f"   Range: [{off_diag_corr.min():.3f}, {off_diag_corr.max():.3f}]")
    
    # ============ FEATURE STATISTICS ============
    print(f"\n5. NORMALIZED FEATURES")
    
    norm_cols = [c for c in full_data.columns if c.endswith('_norm')]
    feature_stats = full_data[norm_cols].describe()
    
    print(f"   Number of normalized features: {len(norm_cols)}")
    print(f"   Mean absolute value: {full_data[norm_cols].abs().mean().mean():.4f}")
    print(f"   Features with values > 3 std: {(full_data[norm_cols].abs() > 3).any().sum()} features")
    
    # Check for missing data
    missing = full_data[norm_cols].isna().sum().sum()
    print(f"   Missing values: {missing} ({missing/len(full_data)/len(norm_cols)*100:.4f}%)")
    
    # ============ MARKET-WIDE STATISTICS ============
    print(f"\n6. MARKET-WIDE CHARACTERISTICS")
    
    market_data = full_data.groupby('date').agg({
        'returns': ['mean', 'std'],
        'close': 'mean'
    })
    
    market_vol = market_data[('returns', 'std')]
    print(f"   Cross-sectional volatility:")
    print(f"   Mean: {market_vol.mean():.4f}")
    print(f"   Range: [{market_vol.min():.4f}, {market_vol.max():.4f}]")
    
    # Volatility clustering (autocorrelation of squared returns)
    market_returns = market_data[('returns', 'mean')]
    acf_lag1 = market_returns.autocorr(lag=1)
    acf_sq_lag1 = (market_returns**2).autocorr(lag=1)
    print(f"   Return autocorr (lag=1): {acf_lag1:.3f}")
    print(f"   Volatility clustering (lag=1): {acf_sq_lag1:.3f}")
    
    print("\n" + "="*80)
    
    return {
        'full_data': full_data,
        'splits': splits,
        'returns_stats': returns_stats,
        'corr_matrix': corr_matrix,
        'feature_stats': feature_stats,
        'returns_pivot': returns_pivot
    }


def create_visualizations(analysis_results, output_dir="dataset_analysis"):
    """Generate comprehensive visualizations."""
    
    Path(output_dir).mkdir(exist_ok=True)
    full_data = analysis_results['full_data']
    returns_pivot = analysis_results['returns_pivot']
    corr_matrix = analysis_results['corr_matrix']
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150
    
    # ============ 1. RETURN DISTRIBUTIONS ============
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Aggregate returns histogram
    all_returns = full_data['returns'].dropna()
    axes[0, 0].hist(all_returns, bins=100, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[0, 0].set_xlabel('Daily Return')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Daily Returns (All Tickers)')
    axes[0, 0].text(0.05, 0.95, f'Skew: {stats.skew(all_returns):.2f}\nKurt: {stats.kurtosis(all_returns):.2f}',
                    transform=axes[0, 0].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # Q-Q plot
    stats.probplot(all_returns, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot vs Normal Distribution')
    
    # Volatility distribution
    vol_by_ticker = full_data.groupby('ticker')['returns'].std()
    axes[1, 0].hist(vol_by_ticker, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Daily Volatility (Std Dev)')
    axes[1, 0].set_ylabel('Number of Tickers')
    axes[1, 0].set_title('Distribution of Ticker Volatilities')
    
    # Cumulative returns by ticker (sample)
    sample_tickers = ['MSFT', 'JPM', 'JNJ', 'XOM', 'WMT']
    for ticker in sample_tickers:
        ticker_data = full_data[full_data['ticker'] == ticker].sort_values('date')
        cumulative = (1 + ticker_data['returns']).cumprod()
        axes[1, 1].plot(ticker_data['date'], cumulative, label=ticker, alpha=0.7)
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Cumulative Return (1 + r)')
    axes[1, 1].set_title('Cumulative Returns (Sample Tickers)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/return_distributions.png")
    plt.close()
    
    # ============ 2. CORRELATION HEATMAP ============
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Return Correlation Matrix (30 Tickers)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()
    
    # ============ 3. FEATURE DISTRIBUTIONS ============
    norm_cols = [c for c in full_data.columns if c.endswith('_norm')]
    sample_features = ['close_norm', 'returns_norm', 'volatility_20d_norm', 
                      'rsi_norm', 'volume_norm', 'macd_norm']
    sample_features = [f for f in sample_features if f in norm_cols]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, feature in enumerate(sample_features[:6]):
        data = full_data[feature].dropna()
        axes[idx].hist(data, bins=50, alpha=0.7, edgecolor='black')
        axes[idx].set_xlabel('Normalized Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(feature.replace('_norm', '').upper())
        axes[idx].axvline(data.mean(), color='red', linestyle='--', linewidth=1)
    
    plt.suptitle('Normalized Feature Distributions', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_distributions.png")
    plt.close()
    
    # ============ 4. TIME SERIES VOLATILITY ============
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Market-wide average return
    market_avg = full_data.groupby('date')['returns'].mean()
    axes[0].plot(market_avg.index, market_avg.values, linewidth=0.8, alpha=0.7)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=0.8)
    axes[0].set_ylabel('Average Return')
    axes[0].set_title('Market-Wide Average Daily Return')
    axes[0].grid(True, alpha=0.3)
    
    # Rolling volatility
    rolling_vol = market_avg.rolling(window=60).std()
    axes[1].plot(rolling_vol.index, rolling_vol.values, linewidth=1, color='orange')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('60-Day Rolling Volatility')
    axes[1].set_title('Market Volatility Over Time')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_series_volatility.png")
    plt.close()
    
    # ============ 5. SECTOR STATISTICS ============
    sector_map = {
        'IBM': 'Tech', 'MSFT': 'Tech', 'ORCL': 'Tech', 'INTC': 'Tech', 'HPQ': 'Tech', 'CSCO': 'Tech',
        'JPM': 'Finance', 'BAC': 'Finance', 'WFC': 'Finance', 'C': 'Finance', 'AXP': 'Finance',
        'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'MRK': 'Healthcare', 'ABT': 'Healthcare',
        'KO': 'Staples', 'PG': 'Staples', 'WMT': 'Staples', 'PEP': 'Staples',
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
        'GE': 'Industrial', 'CAT': 'Industrial', 'BA': 'Industrial',
        'HD': 'Discretionary', 'MCD': 'Discretionary',
        'SO': 'Utilities', 'D': 'Utilities',
        'DD': 'Materials'
    }
    
    full_data['sector'] = full_data['ticker'].map(sector_map)
    sector_stats = full_data.groupby('sector')['returns'].agg(['mean', 'std', 'count'])
    sector_stats['sharpe'] = (sector_stats['mean'] / sector_stats['std']) * np.sqrt(252)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sector returns
    sector_stats['mean'].sort_values().plot(kind='barh', ax=axes[0], color='steelblue')
    axes[0].set_xlabel('Average Daily Return')
    axes[0].set_title('Average Returns by Sector')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Sector Sharpe ratios
    sector_stats['sharpe'].sort_values().plot(kind='barh', ax=axes[1], color='coral')
    axes[1].set_xlabel('Annualized Sharpe Ratio')
    axes[1].set_title('Risk-Adjusted Returns by Sector')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sector_analysis.png")
    plt.close()
    
    print(f"\n✅ Visualizations saved to {output_dir}/")


if __name__ == "__main__":
    # Run analysis
    results = analyze_sp500_dataset()
    
    # Generate visualizations
    create_visualizations(results)
    
    # Save detailed statistics to CSV
    results['returns_stats'].to_csv('dataset_analysis/ticker_statistics.csv')
    results['corr_matrix'].to_csv('dataset_analysis/correlation_matrix.csv')
    
    print("\n✅ Analysis complete!")