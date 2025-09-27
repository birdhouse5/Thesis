import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from environments.data import PortfolioDataset

# Load datasets (force_recreate=True if files don't exist)
sp500_path = Path("environments/data/sp500_rl_ready_cleaned.parquet")
crypto_path = Path("environments/data/crypto_rl_ready_cleaned.parquet")

print("Loading datasets...")
sp500_dataset = PortfolioDataset(
    asset_class="sp500",
    data_path=str(sp500_path),
    force_recreate=not sp500_path.exists()
)

crypto_dataset = PortfolioDataset(
    asset_class="crypto", 
    data_path=str(crypto_path),
    force_recreate=not crypto_path.exists(),
    proportional=True,  # Use proportional splits for crypto
    proportions=(0.7, 0.2, 0.1)
)
print("✅ Datasets loaded")

# Get train/val/test splits
sp500_train = sp500_dataset.get_split("train")
sp500_val = sp500_dataset.get_split("val")
sp500_test = sp500_dataset.get_split("test")

crypto_train = crypto_dataset.get_split("train")
crypto_val = crypto_dataset.get_split("val")
crypto_test = crypto_dataset.get_split("test")

def compute_metrics(test_split, window=20):
    """Compute time series metrics for test split."""
    data = test_split.data
    dates = sorted(data['date'].unique())
    tickers = sorted(data['ticker'].unique())
    
    # Build returns matrix (dates × assets)
    returns_pivot = data.pivot(index='date', columns='ticker', values='returns')
    returns_pivot = returns_pivot.sort_index()
    
    # Build price matrix for average market price
    price_pivot = data.pivot(index='date', columns='ticker', values='close')
    price_pivot = price_pivot.sort_index()
    
    metrics = []
    for i, date in enumerate(dates):
        day_returns = returns_pivot.loc[date].values
        day_prices = price_pivot.loc[date].values
        
        # Average return
        avg_return = np.nanmean(day_returns)
        
        # Cross-sectional volatility (std across assets)
        cross_vol = np.nanstd(day_returns)
        
        # Average market price (sum across all assets)
        avg_price = np.nansum(day_prices)
        
        # Rolling window metrics
        if i >= window:
            # Get historical window
            window_returns = returns_pivot.iloc[i-window:i].values  # (window, n_assets)
            
            # Smoothed average return (rolling mean)
            smoothed_avg_return = np.nanmean(window_returns)
            
            # Smoothed cross-sectional volatility (rolling mean of std)
            smoothed_cross_vol = np.nanmean([np.nanstd(window_returns[j]) for j in range(len(window_returns))])
            
            # Compute correlation matrix
            corr_matrix = np.corrcoef(window_returns.T)  # (n_assets, n_assets)
            
            # Average off-diagonal correlations
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            avg_corr = np.nanmean(corr_matrix[mask])
        else:
            smoothed_avg_return = np.nan
            smoothed_cross_vol = np.nan
            avg_corr = np.nan
            
        metrics.append({
            'date': date,
            'avg_return': avg_return,
            'cross_vol': cross_vol,
            'avg_corr': avg_corr,
            'avg_price': avg_price,
            'smoothed_avg_return': smoothed_avg_return,
            'smoothed_cross_vol': smoothed_cross_vol
        })
    
    return pd.DataFrame(metrics)

# Compute metrics for all splits
sp500_train_metrics = compute_metrics(sp500_train)
sp500_val_metrics = compute_metrics(sp500_val)
sp500_test_metrics = compute_metrics(sp500_test)

crypto_train_metrics = compute_metrics(crypto_train)
crypto_val_metrics = compute_metrics(crypto_val)
crypto_test_metrics = compute_metrics(crypto_test)

# Create time series plots
fig1, axes = plt.subplots(2, 4, figsize=(20, 10))
fig1.suptitle('Test Split Time Series Statistics (20-day Rolling Window)', fontsize=16, y=1.00)

datasets = [
    (sp500_test_metrics, 'SP500', axes[0, :]),
    (crypto_test_metrics, 'Crypto', axes[1, :])
]

for metrics_df, name, ax_row in datasets:
    dates = metrics_df['date']
    
    # Plot 1: Average Return
    ax_row[0].plot(dates, metrics_df['avg_return'], linewidth=1, color='steelblue')
    ax_row[0].axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax_row[0].set_title(f'{name} - Average Return', fontsize=12)
    ax_row[0].set_xlabel('Date')
    ax_row[0].set_ylabel('Return')
    ax_row[0].grid(alpha=0.3)
    ax_row[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Cross-sectional Volatility
    ax_row[1].plot(dates, metrics_df['cross_vol'], linewidth=1, color='darkorange')
    ax_row[1].set_title(f'{name} - Cross-Sectional Volatility', fontsize=12)
    ax_row[1].set_xlabel('Date')
    ax_row[1].set_ylabel('Volatility (Std Dev)')
    ax_row[1].grid(alpha=0.3)
    ax_row[1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Average Correlation
    ax_row[2].plot(dates, metrics_df['avg_corr'], linewidth=1, color='green')
    ax_row[2].set_title(f'{name} - Average Correlation', fontsize=12)
    ax_row[2].set_xlabel('Date')
    ax_row[2].set_ylabel('Correlation')
    ax_row[2].grid(alpha=0.3)
    ax_row[2].tick_params(axis='x', rotation=45)
    
    # Plot 4: Average Market Price (sum across assets)
    ax_row[3].plot(dates, metrics_df['avg_price'], linewidth=1, color='purple')
    ax_row[3].set_title(f'{name} - Total Market Price', fontsize=12)
    ax_row[3].set_xlabel('Date')
    ax_row[3].set_ylabel('Sum of Prices')
    ax_row[3].grid(alpha=0.3)
    ax_row[3].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('test_split_statistics.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved as 'test_split_statistics.png'")

# Create training split time series plots
fig_train, axes_train = plt.subplots(2, 4, figsize=(20, 10))
fig_train.suptitle('Training Split Time Series Statistics (20-day Rolling Window)', fontsize=16, y=1.00)

datasets_train = [
    (sp500_train_metrics, 'SP500', axes_train[0, :]),
    (crypto_train_metrics, 'Crypto', axes_train[1, :])
]

for metrics_df, name, ax_row in datasets_train:
    dates = metrics_df['date']
    
    # Plot 1: Average Return
    ax_row[0].plot(dates, metrics_df['avg_return'], linewidth=1, color='steelblue')
    ax_row[0].axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax_row[0].set_title(f'{name} - Average Return', fontsize=12)
    ax_row[0].set_xlabel('Date')
    ax_row[0].set_ylabel('Return')
    ax_row[0].grid(alpha=0.3)
    ax_row[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Cross-sectional Volatility
    ax_row[1].plot(dates, metrics_df['cross_vol'], linewidth=1, color='darkorange')
    ax_row[1].set_title(f'{name} - Cross-Sectional Volatility', fontsize=12)
    ax_row[1].set_xlabel('Date')
    ax_row[1].set_ylabel('Volatility (Std Dev)')
    ax_row[1].grid(alpha=0.3)
    ax_row[1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Average Correlation
    ax_row[2].plot(dates, metrics_df['avg_corr'], linewidth=1, color='green')
    ax_row[2].set_title(f'{name} - Average Correlation', fontsize=12)
    ax_row[2].set_xlabel('Date')
    ax_row[2].set_ylabel('Correlation')
    ax_row[2].grid(alpha=0.3)
    ax_row[2].tick_params(axis='x', rotation=45)
    
    # Plot 4: Average Market Price (sum across assets)
    ax_row[3].plot(dates, metrics_df['avg_price'], linewidth=1, color='purple')
    ax_row[3].set_title(f'{name} - Total Market Price', fontsize=12)
    ax_row[3].set_xlabel('Date')
    ax_row[3].set_ylabel('Sum of Prices')
    ax_row[3].grid(alpha=0.3)
    ax_row[3].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('train_split_statistics.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved as 'train_split_statistics.png'")

# Create smoothed metrics time series plots
fig_smooth, axes_smooth = plt.subplots(2, 4, figsize=(20, 10))
fig_smooth.suptitle('Smoothed Metrics Time Series (20-day Rolling Window)', fontsize=16, y=1.00)

datasets_smooth = [
    (sp500_test_metrics, 'SP500', axes_smooth[0, :]),
    (crypto_test_metrics, 'Crypto', axes_smooth[1, :])
]

for metrics_df, name, ax_row in datasets_smooth:
    dates = metrics_df['date']
    
    # Plot 1: Smoothed Average Return
    ax_row[0].plot(dates, metrics_df['smoothed_avg_return'], linewidth=1, color='steelblue')
    ax_row[0].axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax_row[0].set_title(f'{name} - Smoothed Avg Return', fontsize=12)
    ax_row[0].set_xlabel('Date')
    ax_row[0].set_ylabel('Return (Rolling)')
    ax_row[0].grid(alpha=0.3)
    ax_row[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Smoothed Cross-sectional Volatility
    ax_row[1].plot(dates, metrics_df['smoothed_cross_vol'], linewidth=1, color='darkorange')
    ax_row[1].set_title(f'{name} - Smoothed Cross-Vol', fontsize=12)
    ax_row[1].set_xlabel('Date')
    ax_row[1].set_ylabel('Volatility (Rolling)')
    ax_row[1].grid(alpha=0.3)
    ax_row[1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Average Correlation (already smoothed)
    ax_row[2].plot(dates, metrics_df['avg_corr'], linewidth=1, color='green')
    ax_row[2].set_title(f'{name} - Average Correlation', fontsize=12)
    ax_row[2].set_xlabel('Date')
    ax_row[2].set_ylabel('Correlation')
    ax_row[2].grid(alpha=0.3)
    ax_row[2].tick_params(axis='x', rotation=45)
    
    # Plot 4: Average Market Price
    ax_row[3].plot(dates, metrics_df['avg_price'], linewidth=1, color='purple')
    ax_row[3].set_title(f'{name} - Total Market Price', fontsize=12)
    ax_row[3].set_xlabel('Date')
    ax_row[3].set_ylabel('Sum of Prices')
    ax_row[3].grid(alpha=0.3)
    ax_row[3].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('smoothed_metrics_statistics.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved as 'smoothed_metrics_statistics.png'")

# Create smoothed training metrics time series plots
fig_smooth_train, axes_smooth_train = plt.subplots(2, 4, figsize=(20, 10))
fig_smooth_train.suptitle('Smoothed Training Metrics Time Series (20-day Rolling Window)', fontsize=16, y=1.00)

datasets_smooth_train = [
    (sp500_train_metrics, 'SP500', axes_smooth_train[0, :]),
    (crypto_train_metrics, 'Crypto', axes_smooth_train[1, :])
]

for metrics_df, name, ax_row in datasets_smooth_train:
    dates = metrics_df['date']
    
    # Plot 1: Smoothed Average Return
    ax_row[0].plot(dates, metrics_df['smoothed_avg_return'], linewidth=1, color='steelblue')
    ax_row[0].axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax_row[0].set_title(f'{name} - Smoothed Avg Return', fontsize=12)
    ax_row[0].set_xlabel('Date')
    ax_row[0].set_ylabel('Return (Rolling)')
    ax_row[0].grid(alpha=0.3)
    ax_row[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Smoothed Cross-sectional Volatility
    ax_row[1].plot(dates, metrics_df['smoothed_cross_vol'], linewidth=1, color='darkorange')
    ax_row[1].set_title(f'{name} - Smoothed Cross-Vol', fontsize=12)
    ax_row[1].set_xlabel('Date')
    ax_row[1].set_ylabel('Volatility (Rolling)')
    ax_row[1].grid(alpha=0.3)
    ax_row[1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Average Correlation (already smoothed)
    ax_row[2].plot(dates, metrics_df['avg_corr'], linewidth=1, color='green')
    ax_row[2].set_title(f'{name} - Average Correlation', fontsize=12)
    ax_row[2].set_xlabel('Date')
    ax_row[2].set_ylabel('Correlation')
    ax_row[2].grid(alpha=0.3)
    ax_row[2].tick_params(axis='x', rotation=45)
    
    # Plot 4: Average Market Price
    ax_row[3].plot(dates, metrics_df['avg_price'], linewidth=1, color='purple')
    ax_row[3].set_title(f'{name} - Total Market Price', fontsize=12)
    ax_row[3].set_xlabel('Date')
    ax_row[3].set_ylabel('Sum of Prices')
    ax_row[3].grid(alpha=0.3)
    ax_row[3].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('smoothed_train_metrics_statistics.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved as 'smoothed_train_metrics_statistics.png'")

# Create smoothed validation metrics time series plots
fig_smooth_val, axes_smooth_val = plt.subplots(2, 4, figsize=(20, 10))
fig_smooth_val.suptitle('Smoothed Validation Metrics Time Series (20-day Rolling Window)', fontsize=16, y=1.00)

datasets_smooth_val = [
    (sp500_val_metrics, 'SP500', axes_smooth_val[0, :]),
    (crypto_val_metrics, 'Crypto', axes_smooth_val[1, :])
]

for metrics_df, name, ax_row in datasets_smooth_val:
    dates = metrics_df['date']
    
    # Plot 1: Smoothed Average Return
    ax_row[0].plot(dates, metrics_df['smoothed_avg_return'], linewidth=1, color='steelblue')
    ax_row[0].axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax_row[0].set_title(f'{name} - Smoothed Avg Return', fontsize=12)
    ax_row[0].set_xlabel('Date')
    ax_row[0].set_ylabel('Return (Rolling)')
    ax_row[0].grid(alpha=0.3)
    ax_row[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Smoothed Cross-sectional Volatility
    ax_row[1].plot(dates, metrics_df['smoothed_cross_vol'], linewidth=1, color='darkorange')
    ax_row[1].set_title(f'{name} - Smoothed Cross-Vol', fontsize=12)
    ax_row[1].set_xlabel('Date')
    ax_row[1].set_ylabel('Volatility (Rolling)')
    ax_row[1].grid(alpha=0.3)
    ax_row[1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Average Correlation (already smoothed)
    ax_row[2].plot(dates, metrics_df['avg_corr'], linewidth=1, color='green')
    ax_row[2].set_title(f'{name} - Average Correlation', fontsize=12)
    ax_row[2].set_xlabel('Date')
    ax_row[2].set_ylabel('Correlation')
    ax_row[2].grid(alpha=0.3)
    ax_row[2].tick_params(axis='x', rotation=45)
    
    # Plot 4: Average Market Price
    ax_row[3].plot(dates, metrics_df['avg_price'], linewidth=1, color='purple')
    ax_row[3].set_title(f'{name} - Total Market Price', fontsize=12)
    ax_row[3].set_xlabel('Date')
    ax_row[3].set_ylabel('Sum of Prices')
    ax_row[3].grid(alpha=0.3)
    ax_row[3].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('smoothed_val_metrics_statistics.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved as 'smoothed_val_metrics_statistics.png'")

# ========== LOG-TRANSFORMED PLOTS ==========

# Log-transformed test split
fig_log_test, axes_log_test = plt.subplots(2, 4, figsize=(20, 10))
fig_log_test.suptitle('Test Split Time Series Statistics - Log Scale (20-day Rolling Window)', fontsize=16, y=1.00)

datasets_log_test = [
    (sp500_test_metrics, 'SP500', axes_log_test[0, :]),
    (crypto_test_metrics, 'Crypto', axes_log_test[1, :])
]

for metrics_df, name, ax_row in datasets_log_test:
    dates = metrics_df['date']
    
    ax_row[0].plot(dates, metrics_df['avg_return'], linewidth=1, color='steelblue')
    ax_row[0].axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax_row[0].set_title(f'{name} - Average Return', fontsize=12)
    ax_row[0].set_xlabel('Date')
    ax_row[0].set_ylabel('Return')
    ax_row[0].set_yscale('symlog')
    ax_row[0].grid(alpha=0.3)
    ax_row[0].tick_params(axis='x', rotation=45)
    
    ax_row[1].plot(dates, metrics_df['cross_vol'], linewidth=1, color='darkorange')
    ax_row[1].set_title(f'{name} - Cross-Sectional Volatility', fontsize=12)
    ax_row[1].set_xlabel('Date')
    ax_row[1].set_ylabel('Volatility (Std Dev)')
    ax_row[1].set_yscale('log')
    ax_row[1].grid(alpha=0.3)
    ax_row[1].tick_params(axis='x', rotation=45)
    
    ax_row[2].plot(dates, metrics_df['avg_corr'], linewidth=1, color='green')
    ax_row[2].set_title(f'{name} - Average Correlation', fontsize=12)
    ax_row[2].set_xlabel('Date')
    ax_row[2].set_ylabel('Correlation')
    ax_row[2].set_yscale('symlog')
    ax_row[2].grid(alpha=0.3)
    ax_row[2].tick_params(axis='x', rotation=45)
    
    ax_row[3].plot(dates, metrics_df['avg_price'], linewidth=1, color='purple')
    ax_row[3].set_title(f'{name} - Total Market Price', fontsize=12)
    ax_row[3].set_xlabel('Date')
    ax_row[3].set_ylabel('Sum of Prices')
    ax_row[3].set_yscale('log')
    ax_row[3].grid(alpha=0.3)
    ax_row[3].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('log_test_split_statistics.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved as 'log_test_split_statistics.png'")

# Log-transformed training split
fig_log_train, axes_log_train = plt.subplots(2, 4, figsize=(20, 10))
fig_log_train.suptitle('Training Split Time Series Statistics - Log Scale (20-day Rolling Window)', fontsize=16, y=1.00)

datasets_log_train = [
    (sp500_train_metrics, 'SP500', axes_log_train[0, :]),
    (crypto_train_metrics, 'Crypto', axes_log_train[1, :])
]

for metrics_df, name, ax_row in datasets_log_train:
    dates = metrics_df['date']
    
    ax_row[0].plot(dates, metrics_df['avg_return'], linewidth=1, color='steelblue')
    ax_row[0].axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax_row[0].set_title(f'{name} - Average Return', fontsize=12)
    ax_row[0].set_xlabel('Date')
    ax_row[0].set_ylabel('Return')
    ax_row[0].set_yscale('symlog')
    ax_row[0].grid(alpha=0.3)
    ax_row[0].tick_params(axis='x', rotation=45)
    
    ax_row[1].plot(dates, metrics_df['cross_vol'], linewidth=1, color='darkorange')
    ax_row[1].set_title(f'{name} - Cross-Sectional Volatility', fontsize=12)
    ax_row[1].set_xlabel('Date')
    ax_row[1].set_ylabel('Volatility (Std Dev)')
    ax_row[1].set_yscale('log')
    ax_row[1].grid(alpha=0.3)
    ax_row[1].tick_params(axis='x', rotation=45)
    
    ax_row[2].plot(dates, metrics_df['avg_corr'], linewidth=1, color='green')
    ax_row[2].set_title(f'{name} - Average Correlation', fontsize=12)
    ax_row[2].set_xlabel('Date')
    ax_row[2].set_ylabel('Correlation')
    ax_row[2].set_yscale('symlog')
    ax_row[2].grid(alpha=0.3)
    ax_row[2].tick_params(axis='x', rotation=45)
    
    ax_row[3].plot(dates, metrics_df['avg_price'], linewidth=1, color='purple')
    ax_row[3].set_title(f'{name} - Total Market Price', fontsize=12)
    ax_row[3].set_xlabel('Date')
    ax_row[3].set_ylabel('Sum of Prices')
    ax_row[3].set_yscale('log')
    ax_row[3].grid(alpha=0.3)
    ax_row[3].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('log_train_split_statistics.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved as 'log_train_split_statistics.png'")

# Log-transformed smoothed test metrics
fig_log_smooth, axes_log_smooth = plt.subplots(2, 4, figsize=(20, 10))
fig_log_smooth.suptitle('Smoothed Metrics Time Series - Log Scale (20-day Rolling Window)', fontsize=16, y=1.00)

datasets_log_smooth = [
    (sp500_test_metrics, 'SP500', axes_log_smooth[0, :]),
    (crypto_test_metrics, 'Crypto', axes_log_smooth[1, :])
]

for metrics_df, name, ax_row in datasets_log_smooth:
    dates = metrics_df['date']
    
    ax_row[0].plot(dates, metrics_df['smoothed_avg_return'], linewidth=1, color='steelblue')
    ax_row[0].axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax_row[0].set_title(f'{name} - Smoothed Avg Return', fontsize=12)
    ax_row[0].set_xlabel('Date')
    ax_row[0].set_ylabel('Return (Rolling)')
    ax_row[0].set_yscale('symlog')
    ax_row[0].grid(alpha=0.3)
    ax_row[0].tick_params(axis='x', rotation=45)
    
    ax_row[1].plot(dates, metrics_df['smoothed_cross_vol'], linewidth=1, color='darkorange')
    ax_row[1].set_title(f'{name} - Smoothed Cross-Vol', fontsize=12)
    ax_row[1].set_xlabel('Date')
    ax_row[1].set_ylabel('Volatility (Rolling)')
    ax_row[1].set_yscale('log')
    ax_row[1].grid(alpha=0.3)
    ax_row[1].tick_params(axis='x', rotation=45)
    
    ax_row[2].plot(dates, metrics_df['avg_corr'], linewidth=1, color='green')
    ax_row[2].set_title(f'{name} - Average Correlation', fontsize=12)
    ax_row[2].set_xlabel('Date')
    ax_row[2].set_ylabel('Correlation')
    ax_row[2].set_yscale('symlog')
    ax_row[2].grid(alpha=0.3)
    ax_row[2].tick_params(axis='x', rotation=45)
    
    ax_row[3].plot(dates, metrics_df['avg_price'], linewidth=1, color='purple')
    ax_row[3].set_title(f'{name} - Total Market Price', fontsize=12)
    ax_row[3].set_xlabel('Date')
    ax_row[3].set_ylabel('Sum of Prices')
    ax_row[3].set_yscale('log')
    ax_row[3].grid(alpha=0.3)
    ax_row[3].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('log_smoothed_metrics_statistics.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved as 'log_smoothed_metrics_statistics.png'")

# Log-transformed smoothed training metrics
fig_log_smooth_train, axes_log_smooth_train = plt.subplots(2, 4, figsize=(20, 10))
fig_log_smooth_train.suptitle('Smoothed Training Metrics Time Series - Log Scale (20-day Rolling Window)', fontsize=16, y=1.00)

datasets_log_smooth_train = [
    (sp500_train_metrics, 'SP500', axes_log_smooth_train[0, :]),
    (crypto_train_metrics, 'Crypto', axes_log_smooth_train[1, :])
]

for metrics_df, name, ax_row in datasets_log_smooth_train:
    dates = metrics_df['date']
    
    ax_row[0].plot(dates, metrics_df['smoothed_avg_return'], linewidth=1, color='steelblue')
    ax_row[0].axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax_row[0].set_title(f'{name} - Smoothed Avg Return', fontsize=12)
    ax_row[0].set_xlabel('Date')
    ax_row[0].set_ylabel('Return (Rolling)')
    ax_row[0].set_yscale('symlog')
    ax_row[0].grid(alpha=0.3)
    ax_row[0].tick_params(axis='x', rotation=45)
    
    ax_row[1].plot(dates, metrics_df['smoothed_cross_vol'], linewidth=1, color='darkorange')
    ax_row[1].set_title(f'{name} - Smoothed Cross-Vol', fontsize=12)
    ax_row[1].set_xlabel('Date')
    ax_row[1].set_ylabel('Volatility (Rolling)')
    ax_row[1].set_yscale('log')
    ax_row[1].grid(alpha=0.3)
    ax_row[1].tick_params(axis='x', rotation=45)
    
    ax_row[2].plot(dates, metrics_df['avg_corr'], linewidth=1, color='green')
    ax_row[2].set_title(f'{name} - Average Correlation', fontsize=12)
    ax_row[2].set_xlabel('Date')
    ax_row[2].set_ylabel('Correlation')
    ax_row[2].set_yscale('symlog')
    ax_row[2].grid(alpha=0.3)
    ax_row[2].tick_params(axis='x', rotation=45)
    
    ax_row[3].plot(dates, metrics_df['avg_price'], linewidth=1, color='purple')
    ax_row[3].set_title(f'{name} - Total Market Price', fontsize=12)
    ax_row[3].set_xlabel('Date')
    ax_row[3].set_ylabel('Sum of Prices')
    ax_row[3].set_yscale('log')
    ax_row[3].grid(alpha=0.3)
    ax_row[3].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('log_smoothed_train_metrics_statistics.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved as 'log_smoothed_train_metrics_statistics.png'")

# Log-transformed smoothed validation metrics
fig_log_smooth_val, axes_log_smooth_val = plt.subplots(2, 4, figsize=(20, 10))
fig_log_smooth_val.suptitle('Smoothed Validation Metrics Time Series - Log Scale (20-day Rolling Window)', fontsize=16, y=1.00)

datasets_log_smooth_val = [
    (sp500_val_metrics, 'SP500', axes_log_smooth_val[0, :]),
    (crypto_val_metrics, 'Crypto', axes_log_smooth_val[1, :])
]

for metrics_df, name, ax_row in datasets_log_smooth_val:
    dates = metrics_df['date']
    
    ax_row[0].plot(dates, metrics_df['smoothed_avg_return'], linewidth=1, color='steelblue')
    ax_row[0].axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax_row[0].set_title(f'{name} - Smoothed Avg Return', fontsize=12)
    ax_row[0].set_xlabel('Date')
    ax_row[0].set_ylabel('Return (Rolling)')
    ax_row[0].set_yscale('symlog')
    ax_row[0].grid(alpha=0.3)
    ax_row[0].tick_params(axis='x', rotation=45)
    
    ax_row[1].plot(dates, metrics_df['smoothed_cross_vol'], linewidth=1, color='darkorange')
    ax_row[1].set_title(f'{name} - Smoothed Cross-Vol', fontsize=12)
    ax_row[1].set_xlabel('Date')
    ax_row[1].set_ylabel('Volatility (Rolling)')
    ax_row[1].set_yscale('log')
    ax_row[1].grid(alpha=0.3)
    ax_row[1].tick_params(axis='x', rotation=45)
    
    ax_row[2].plot(dates, metrics_df['avg_corr'], linewidth=1, color='green')
    ax_row[2].set_title(f'{name} - Average Correlation', fontsize=12)
    ax_row[2].set_xlabel('Date')
    ax_row[2].set_ylabel('Correlation')
    ax_row[2].set_yscale('symlog')
    ax_row[2].grid(alpha=0.3)
    ax_row[2].tick_params(axis='x', rotation=45)
    
    ax_row[3].plot(dates, metrics_df['avg_price'], linewidth=1, color='purple')
    ax_row[3].set_title(f'{name} - Total Market Price', fontsize=12)
    ax_row[3].set_xlabel('Date')
    ax_row[3].set_ylabel('Sum of Prices')
    ax_row[3].set_yscale('log')
    ax_row[3].grid(alpha=0.3)
    ax_row[3].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('log_smoothed_val_metrics_statistics.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved as 'log_smoothed_val_metrics_statistics.png'")

# Create distribution comparison plots
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
fig2.suptitle('Distribution Comparison Across Splits', fontsize=16, y=1.00)

# SP500 distributions
sp500_data = [
    (sp500_train_metrics, sp500_val_metrics, sp500_test_metrics, 'SP500', axes2[0, :])
]
crypto_data = [
    (crypto_train_metrics, crypto_val_metrics, crypto_test_metrics, 'Crypto', axes2[1, :])
]

for train_m, val_m, test_m, name, ax_row in sp500_data + crypto_data:
    # Plot 1: Average Return Distribution
    ax_row[0].hist(train_m['avg_return'].dropna(), bins=30, alpha=0.5, label='Train', color='blue')
    ax_row[0].hist(val_m['avg_return'].dropna(), bins=30, alpha=0.5, label='Val', color='orange')
    ax_row[0].hist(test_m['avg_return'].dropna(), bins=30, alpha=0.5, label='Test', color='green')
    ax_row[0].set_title(f'{name} - Avg Return Distribution', fontsize=12)
    ax_row[0].set_xlabel('Return')
    ax_row[0].set_ylabel('Frequency')
    ax_row[0].legend()
    ax_row[0].grid(alpha=0.3)
    
    # Plot 2: Cross-sectional Volatility Distribution
    ax_row[1].hist(train_m['cross_vol'].dropna(), bins=30, alpha=0.5, label='Train', color='blue')
    ax_row[1].hist(val_m['cross_vol'].dropna(), bins=30, alpha=0.5, label='Val', color='orange')
    ax_row[1].hist(test_m['cross_vol'].dropna(), bins=30, alpha=0.5, label='Test', color='green')
    ax_row[1].set_title(f'{name} - Cross-Vol Distribution', fontsize=12)
    ax_row[1].set_xlabel('Volatility')
    ax_row[1].set_ylabel('Frequency')
    ax_row[1].legend()
    ax_row[1].grid(alpha=0.3)
    
    # Plot 3: Average Correlation Distribution
    ax_row[2].hist(train_m['avg_corr'].dropna(), bins=30, alpha=0.5, label='Train', color='blue')
    ax_row[2].hist(val_m['avg_corr'].dropna(), bins=30, alpha=0.5, label='Val', color='orange')
    ax_row[2].hist(test_m['avg_corr'].dropna(), bins=30, alpha=0.5, label='Test', color='green')
    ax_row[2].set_title(f'{name} - Avg Correlation Distribution', fontsize=12)
    ax_row[2].set_xlabel('Correlation')
    ax_row[2].set_ylabel('Frequency')
    ax_row[2].legend()
    ax_row[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('split_distributions.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved as 'split_distributions.png'")

# Print summary statistics
print("\n=== SP500 Summary ===")
for split_name, metrics_df in [('Train', sp500_train_metrics), ('Val', sp500_val_metrics), ('Test', sp500_test_metrics)]:
    print(f"\n{split_name} Split:")
    print(f"  Date range: {metrics_df['date'].min().date()} to {metrics_df['date'].max().date()}")
    print(f"  Avg return: {metrics_df['avg_return'].mean():.4f} ± {metrics_df['avg_return'].std():.4f}")
    print(f"  Avg cross-vol: {metrics_df['cross_vol'].mean():.4f} ± {metrics_df['cross_vol'].std():.4f}")
    print(f"  Avg correlation: {metrics_df['avg_corr'].mean():.4f} ± {metrics_df['avg_corr'].std():.4f}")
    print(f"  Market price range: {metrics_df['avg_price'].min():.2f} to {metrics_df['avg_price'].max():.2f}")

print("\n=== Crypto Summary ===")
for split_name, metrics_df in [('Train', crypto_train_metrics), ('Val', crypto_val_metrics), ('Test', crypto_test_metrics)]:
    print(f"\n{split_name} Split:")
    print(f"  Date range: {metrics_df['date'].min().date()} to {metrics_df['date'].max().date()}")
    print(f"  Avg return: {metrics_df['avg_return'].mean():.4f} ± {metrics_df['avg_return'].std():.4f}")
    print(f"  Avg cross-vol: {metrics_df['cross_vol'].mean():.4f} ± {metrics_df['cross_vol'].std():.4f}")
    print(f"  Avg correlation: {metrics_df['avg_corr'].mean():.4f} ± {metrics_df['avg_corr'].std():.4f}")
    print(f"  Market price range: {metrics_df['avg_price'].min():.2f} to {metrics_df['avg_price'].max():.2f}")