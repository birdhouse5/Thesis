# data_validator.py
import pandas as pd
import numpy as np

def validate_and_clean_data(input_path, output_path):
    """Ensure data is in exact format: T x N x F with no missing values"""
    
    print("Loading data...")
    df = pd.read_parquet(input_path)
    
    # Basic info
    print(f"Original shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Tickers: {sorted(df['ticker'].unique())}")
    
    # 1. Check for missing dates per ticker
    print("\nChecking for missing data...")
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker]
        ticker_dates = set(ticker_data['date'])
        missing_count = len(set(df['date'].unique()) - ticker_dates)
        if missing_count > 0:
            print(f"Ticker {ticker} missing {missing_count} dates")
    
    # 2. Ensure each date has exactly 30 tickers
    date_counts = df.groupby('date').size()
    irregular_dates = date_counts[date_counts != 30]
    if len(irregular_dates) > 0:
        print(f"Found {len(irregular_dates)} dates without exactly 30 tickers")
        print("Removing these dates...")
        valid_dates = date_counts[date_counts == 30].index
        df = df[df['date'].isin(valid_dates)]
    
    # 3. Check for NaN values
    nan_counts = df.isnull().sum()
    if nan_counts.sum() > 0:
        print("NaN values found:")
        print(nan_counts[nan_counts > 0])
        # Option: drop or fill
        df = df.dropna()
    
    print(f"\nFinal shape: {df.shape}")
    print(f"Final dates: {len(df['date'].unique())}")
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    
    return df

# Usage
if __name__ == "__main__":
    cleaned_df = validate_and_clean_data("environments/data/sp500_rl_ready_cleaned.parquet", "environments/data/sp500_rl_ready_cleaned_2.parquet")