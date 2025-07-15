"""
Targeted NaN cleaning for sp500_rl_ready.parquet based on diagnosis.
We know exactly what we're dealing with: warmup period issues.
"""

import pandas as pd
import numpy as np

def clean_warmup_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean NaNs that are caused by warmup periods in technical indicators.
    Based on diagnosis: we have systematic warmup issues, not data quality problems.
    """
    print("=== TARGETED NaN CLEANING ===")
    print("Strategy: Handle warmup period NaNs with appropriate methods")
    
    df_cleaned = df.copy()
    cleaning_log = {}
    
    # Get features with NaNs (we know which ones from diagnosis)
    features_with_nans = ['BB_Upper_norm', 'BB_Lower_norm', 'RSI_norm', 'CCI_norm', 
                         'returns', 'log_returns', 'excess_returns', 'market_return',
                         'ROC_norm', 'volatility_5d', 'volatility_20d']
    
    print(f"Cleaning {len(features_with_nans)} features with warmup NaNs...")
    
    for feature in features_with_nans:
        if feature not in df.columns:
            continue
            
        original_nans = df_cleaned[feature].isnull().sum()
        
        if feature in ['returns', 'log_returns']:
            # Returns: First day should be 0 (no previous day to compare)
            df_cleaned[feature] = df_cleaned.groupby('ticker')[feature].fillna(0.0)
            strategy = "fill_with_zero"
            
        elif feature == 'excess_returns':
            # Excess returns: Also 0 on first day
            df_cleaned[feature] = df_cleaned.groupby('ticker')[feature].fillna(0.0)
            strategy = "fill_with_zero"
            
        elif feature == 'market_return':
            # Market return: 0 on first day
            df_cleaned[feature] = df_cleaned[feature].fillna(0.0)
            strategy = "fill_with_zero"
            
        elif feature == 'RSI_norm':
            # RSI: Neutral value is 0.5 (since it's normalized from 50)
            df_cleaned[feature] = df_cleaned.groupby('ticker')[feature].fillna(0.5)
            strategy = "fill_with_neutral"
            
        elif feature == 'CCI_norm':
            # CCI: Normalized around 0, so fill with 0
            df_cleaned[feature] = df_cleaned.groupby('ticker')[feature].fillna(0.0)
            strategy = "fill_with_neutral"
            
        elif feature in ['volatility_5d', 'volatility_20d']:
            # Volatility: Use first available value (carry backward)
            df_cleaned[feature] = df_cleaned.groupby('ticker')[feature].fillna(method='bfill')
            strategy = "backward_fill"
            
        elif feature == 'ROC_norm':
            # Rate of Change: 0 for first periods (no change)
            df_cleaned[feature] = df_cleaned.groupby('ticker')[feature].fillna(0.0)
            strategy = "fill_with_zero"
            
        else:
            # Bollinger Bands and other indicators: forward fill within ticker
            df_cleaned[feature] = df_cleaned.groupby('ticker')[feature].fillna(method='bfill')
            strategy = "backward_fill"
        
        final_nans = df_cleaned[feature].isnull().sum()
        cleaning_log[feature] = {
            'original_nans': original_nans,
            'final_nans': final_nans,
            'strategy': strategy,
            'removed': original_nans - final_nans
        }
        
        print(f"  {feature:20}: {original_nans:3d} → {final_nans:3d} NaNs ({strategy})")
    
    return df_cleaned, cleaning_log

def verify_cleaning(df_original: pd.DataFrame, df_cleaned: pd.DataFrame, cleaning_log: dict):
    """Verify that cleaning worked as expected."""
    print(f"\n=== CLEANING VERIFICATION ===")
    
    original_nans = df_original.isnull().sum().sum()
    final_nans = df_cleaned.isnull().sum().sum()
    
    print(f"Total NaNs: {original_nans:,} → {final_nans:,}")
    print(f"Removed: {original_nans - final_nans:,} NaNs")
    
    if final_nans == 0:
        print("✅ Perfect! All NaNs removed")
    else:
        print(f"⚠️  Still have {final_nans} NaNs remaining")
        remaining = df_cleaned.isnull().sum()
        remaining = remaining[remaining > 0]
        for feature, count in remaining.items():
            print(f"    {feature}: {count} NaNs")
    
    # Check data integrity - make sure we didn't break anything
    print(f"\n=== DATA INTEGRITY CHECK ===")
    
    # Check if we have reasonable value ranges
    sample_features = ['returns', 'volatility_5d', 'RSI_norm']
    for feature in sample_features:
        if feature in df_cleaned.columns:
            values = df_cleaned[feature].dropna()
            print(f"{feature:15}: range [{values.min():.6f}, {values.max():.6f}], mean {values.mean():.6f}")
    
    # Check first few dates to see if warmup period looks reasonable
    print(f"\nFirst 5 trading days sample (ABT ticker):")
    abt_early = df_cleaned[df_cleaned['ticker'] == 'ABT'].head(5)
    for _, row in abt_early.iterrows():
        print(f"  {row['date'].date()}: returns={row['returns']:.6f}, RSI={row.get('RSI_norm', 'N/A'):.6f}")

def save_cleaned_data(df_cleaned: pd.DataFrame, output_path: str):
    """Save the cleaned dataset."""
    print(f"\n=== SAVING CLEANED DATA ===")
    
    # Save to parquet
    df_cleaned.to_parquet(output_path, index=False)
    
    # Get file size
    import os
    file_size = os.path.getsize(output_path) / 1024**2
    
    print(f"✅ Saved to: {output_path}")
    print(f"File size: {file_size:.1f} MB")
    print(f"Shape: {df_cleaned.shape}")
    
    # Quick final check
    final_nans = df_cleaned.isnull().sum().sum()
    if final_nans == 0:
        print("✅ Confirmed: No NaNs in saved file")
    else:
        print(f"⚠️  Warning: {final_nans} NaNs still present")

def main_cleaning(input_path: str = '../data/sp500_rl_ready.parquet', 
                  output_path: str = '../data/sp500_rl_ready_cleaned.parquet'):
    """
    Main cleaning function - load, clean, verify, save.
    """
    print("SP500 RL-READY DATA - TARGETED CLEANING")
    print("=" * 50)
    
    # Load data
    print(f"Loading: {input_path}")
    df_original = pd.read_parquet(input_path)
    print(f"Original shape: {df_original.shape}")
    print(f"Original NaNs: {df_original.isnull().sum().sum():,}")
    
    # Clean NaNs
    df_cleaned, cleaning_log = clean_warmup_nans(df_original)
    
    # Verify cleaning
    verify_cleaning(df_original, df_cleaned, cleaning_log)
    
    # Save cleaned data
    save_cleaned_data(df_cleaned, output_path)
    
    print(f"\n🎉 CLEANING COMPLETE!")
    print(f"Ready for train-test split and MDP construction")
    
    return df_cleaned

if __name__ == "__main__":
    # Run the cleaning
    df_cleaned = main_cleaning()