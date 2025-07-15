"""
Focused NaN diagnosis for sp500_rl_ready.parquet
Let's understand what we're dealing with before implementing any fixes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_basic_info(file_path: str = '../data/sp500_rl_ready.parquet'):
    """Load data and show basic information."""
    print("=== LOADING DATA ===")
    df = pd.read_parquet(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of tickers: {df['ticker'].nunique()}")
    print(f"Number of trading days: {df['date'].nunique()}")
    
    # Show first few rows
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    
    # Show column names grouped by type
    print(f"\nColumn structure ({len(df.columns)} total):")
    
    # Basic columns
    basic_cols = ['date', 'ticker']
    print(f"  Basic: {basic_cols}")
    
    # Normalized features  
    norm_cols = [col for col in df.columns if col.endswith('_norm')]
    print(f"  Normalized features: {len(norm_cols)} columns")
    if len(norm_cols) <= 10:
        print(f"    {norm_cols}")
    else:
        print(f"    {norm_cols[:5]} ... {norm_cols[-3:]}")
    
    # Other features
    other_cols = [col for col in df.columns if col not in basic_cols and not col.endswith('_norm')]
    print(f"  Other features: {len(other_cols)} columns")
    print(f"    {other_cols}")
    
    return df

def diagnose_nan_overview(df: pd.DataFrame):
    """Get high-level overview of NaN situation."""
    print(f"\n=== NaN OVERVIEW ===")
    
    total_values = df.shape[0] * df.shape[1]
    total_nans = df.isnull().sum().sum()
    nan_percentage = (total_nans / total_values) * 100
    
    print(f"Total values in dataset: {total_values:,}")
    print(f"Total NaN values: {total_nans:,} ({nan_percentage:.2f}%)")
    
    # Count features with NaNs
    features_with_nans = df.isnull().sum()
    features_with_nans = features_with_nans[features_with_nans > 0]
    
    print(f"Features with NaNs: {len(features_with_nans)}/{len(df.columns)}")
    
    if len(features_with_nans) == 0:
        print("✅ No NaNs found!")
        return
    
    # Show top problematic features
    print(f"\nTop 10 features with most NaNs:")
    for feature in features_with_nans.sort_values(ascending=False).head(10).index:
        count = features_with_nans[feature]
        pct = (count / len(df)) * 100
        print(f"  {feature:25} : {count:6,} ({pct:5.2f}%)")

def diagnose_nan_by_time(df: pd.DataFrame):
    """Analyze NaN patterns over time."""
    print(f"\n=== NaN PATTERNS BY TIME ===")
    
    # Count NaNs per date
    nan_by_date = df.groupby('date').apply(lambda x: x.isnull().sum().sum())
    
    print(f"Dates with NaNs: {(nan_by_date > 0).sum()}/{len(nan_by_date)}")
    print(f"Max NaNs on single date: {nan_by_date.max()}")
    print(f"Mean NaNs per date: {nan_by_date.mean():.1f}")
    
    # Show dates with most NaNs
    if nan_by_date.max() > 0:
        worst_dates = nan_by_date.sort_values(ascending=False).head(5)
        print(f"\nDates with most NaNs:")
        for date, count in worst_dates.items():
            print(f"  {date.date()}: {count:,} NaNs")
        
        # Check if early dates are problematic
        earliest_dates = sorted(df['date'].unique())[:10]
        early_nan_counts = [nan_by_date[date] for date in earliest_dates]
        
        print(f"\nFirst 10 dates NaN counts:")
        for date, count in zip(earliest_dates, early_nan_counts):
            print(f"  {date.date()}: {count:,} NaNs")
        
        if sum(early_nan_counts) > sum(early_nan_counts[-5:]):
            print("🔍 PATTERN: More NaNs in early dates (likely warmup period issue)")

def diagnose_nan_by_ticker(df: pd.DataFrame):
    """Analyze NaN patterns by ticker."""
    print(f"\n=== NaN PATTERNS BY TICKER ===")
    
    # Count NaNs per ticker
    nan_by_ticker = df.groupby('ticker').apply(lambda x: x.isnull().sum().sum())
    
    print(f"Tickers with NaNs: {(nan_by_ticker > 0).sum()}/{len(nan_by_ticker)}")
    print(f"Max NaNs for single ticker: {nan_by_ticker.max()}")
    print(f"Mean NaNs per ticker: {nan_by_ticker.mean():.1f}")
    
    if nan_by_ticker.max() > 0:
        worst_tickers = nan_by_ticker.sort_values(ascending=False).head(5)
        print(f"\nTickers with most NaNs:")
        for ticker, count in worst_tickers.items():
            ticker_total_values = len(df[df['ticker'] == ticker]) * len(df.columns)
            pct = (count / ticker_total_values) * 100
            print(f"  {ticker:6}: {count:6,} NaNs ({pct:5.2f}% of ticker's data)")

def diagnose_nan_by_feature_type(df: pd.DataFrame):
    """Analyze NaN patterns by feature categories."""
    print(f"\n=== NaN PATTERNS BY FEATURE TYPE ===")
    
    # Categorize features
    feature_categories = {
        'Basic': ['date', 'ticker'],
        'Price (normalized)': [],
        'Technical indicators': [],
        'Volume': [],
        'Returns': [],
        'Other': []
    }
    
    for col in df.columns:
        if col in ['date', 'ticker']:
            continue  # Already in Basic
        elif any(x in col.lower() for x in ['open_norm', 'high_norm', 'low_norm', 'close_norm', 'vwap_norm']):
            feature_categories['Price (normalized)'].append(col)
        elif any(x in col.lower() for x in ['rsi', 'macd', 'bb_', 'williams', 'stoch', 'aroon', 'cci', 'cmo', 'mfi', 'atr', 'natr']):
            feature_categories['Technical indicators'].append(col)
        elif 'volume' in col.lower():
            feature_categories['Volume'].append(col)
        elif 'return' in col.lower():
            feature_categories['Returns'].append(col)
        else:
            feature_categories['Other'].append(col)
    
    # Analyze NaNs by category
    for category, features in feature_categories.items():
        if not features:
            continue
            
        category_nans = df[features].isnull().sum().sum()
        category_total = len(df) * len(features)
        category_pct = (category_nans / category_total) * 100
        
        features_with_nans = [f for f in features if df[f].isnull().any()]
        
        print(f"\n{category}:")
        print(f"  Features: {len(features)}")
        print(f"  Features with NaNs: {len(features_with_nans)}")
        print(f"  Total NaNs: {category_nans:,} ({category_pct:.2f}%)")
        
        if features_with_nans:
            print(f"  Problematic features:")
            for feature in features_with_nans:
                nan_count = df[feature].isnull().sum()
                nan_pct = (nan_count / len(df)) * 100
                print(f"    {feature:30}: {nan_count:6,} ({nan_pct:5.2f}%)")

def diagnose_specific_examples(df: pd.DataFrame):
    """Look at specific examples to understand NaN patterns."""
    print(f"\n=== SPECIFIC EXAMPLES ===")
    
    # Find features with NaNs
    features_with_nans = df.columns[df.isnull().any()].tolist()
    
    if not features_with_nans:
        print("No NaNs to examine!")
        return
    
    # Take first feature with NaNs as example
    example_feature = features_with_nans[0]
    
    print(f"Examining feature: {example_feature}")
    print(f"Total NaNs: {df[example_feature].isnull().sum()}")
    
    # Show first few rows of this feature
    print(f"\nFirst 10 values:")
    first_10 = df[['date', 'ticker', example_feature]].head(10)
    for _, row in first_10.iterrows():
        nan_status = "NaN" if pd.isna(row[example_feature]) else f"{row[example_feature]:.6f}"
        print(f"  {row['date'].date()} {row['ticker']:6}: {nan_status}")
    
    # Check if NaNs are at the beginning
    first_non_nan_idx = df[example_feature].first_valid_index()
    if first_non_nan_idx is not None and first_non_nan_idx > 0:
        print(f"\n🔍 First non-NaN value at index {first_non_nan_idx}")
        print("This suggests warmup period issue!")
    
    # Show some non-NaN values for context
    non_nan_values = df[example_feature].dropna().head(5)
    if len(non_nan_values) > 0:
        print(f"\nFirst 5 non-NaN values: {non_nan_values.tolist()}")
        print(f"Range: {non_nan_values.min():.6f} to {non_nan_values.max():.6f}")

def main_diagnosis(file_path: str = '../data/sp500_rl_ready.parquet'):
    """Run complete NaN diagnosis."""
    print("SP500 RL-READY DATA - NaN DIAGNOSIS")
    print("=" * 50)
    
    # Load data
    df = load_and_basic_info(file_path)
    
    # Run diagnostic steps
    diagnose_nan_overview(df)
    diagnose_nan_by_time(df)
    diagnose_nan_by_ticker(df)
    diagnose_nan_by_feature_type(df)
    diagnose_specific_examples(df)
    
    print(f"\n=== DIAGNOSIS COMPLETE ===")
    print("Next step: Based on these findings, decide on cleaning strategy")
    
    return df

if __name__ == "__main__":
    # Run the diagnosis
    df = main_diagnosis()