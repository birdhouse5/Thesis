# loader.py
"""
Minimal S&P 500 historical data loader.
Single file module that can be easily integrated into larger projects.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_sp500_history(
    constituents_file='data/sp500_constituents.csv',
    start_date='1987-01-01',
    end_date='2025-01-01',
    output_file='data/sp500_dataset.parquet'
):
    """
    Main function to load S&P 500 historical data.
    
    Args:
        constituents_file: Path to CSV with columns [ticker, start_date, end_date]
        start_date: Global start date for dataset
        end_date: Global end date for dataset
        output_file: Where to save the final dataset
        
    Returns:
        pandas.DataFrame: Multi-index DataFrame (date, ticker) with price data
    """
    # Create data directory if needed
    Path(output_file).parent.mkdir(exist_ok=True)
    
    # Load constituents
    logger.info(f"Loading constituents from {constituents_file}")
    members = pd.read_csv(constituents_file)
    members['start_date'] = pd.to_datetime(members['start_date'])
    members['end_date'] = pd.to_datetime(members['end_date'])
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    global_start = pd.to_datetime(start_date)
    global_end = pd.to_datetime(end_date)
    
    # Get unique tickers
    tickers = members['ticker'].unique()
    logger.info(f"Found {len(tickers)} unique tickers")
    
    # Download and process each ticker
    all_data = {}
    successful_downloads = 0
    
    for i, ticker in enumerate(tickers):
        if i % 10 == 0:
            logger.info(f"Processing {i}/{len(tickers)} tickers...")
        
        # Get membership periods for this ticker
        ticker_periods = members[members['ticker'] == ticker]
        
        # Initialize empty data
        ticker_data = pd.DataFrame(index=dates)
        ticker_data['adj_close'] = np.nan
        ticker_data['volume'] = np.nan
        ticker_data['is_active'] = 0
        
        # Track if we got any data for this ticker
        got_data = False
        
        # Download data for each membership period
        for _, period in ticker_periods.iterrows():
            try:
                # Determine period boundaries
                period_start = period['start_date']
                period_end = period['end_date'] if pd.notna(period['end_date']) else pd.to_datetime(end_date)
                
                # Download data
                data = yf.download(
                    ticker, 
                    start=period_start, 
                    end=period_end,
                    progress=False,
                    auto_adjust=False  # Explicitly request unadjusted data to get Adj Close column
                )
                
                if not data.empty:
                    logger.debug(f"Downloaded {len(data)} rows for {ticker}")
                    logger.debug(f"Columns available: {data.columns.tolist()}")
                    logger.debug(f"Date range: {data.index[0]} to {data.index[-1]}")
                    
                    # Filter to only include dates within our global range
                    data = data[(data.index >= global_start) & 
                               (data.index <= global_end)]
                    
                    if data.empty:
                        logger.warning(f"No data for {ticker} after filtering to global date range")
                        continue
                    
                    # Only update for dates that exist in both indices
                    common_dates = ticker_data.index.intersection(data.index)
                    
                    logger.debug(f"Common dates found: {len(common_dates)}")
                    
                    if len(common_dates) > 0:
                        # Handle multi-level columns from yfinance
                        if isinstance(data.columns, pd.MultiIndex):
                            # Multi-level columns: ('Adj Close', 'AAPL')
                            adj_close_col = ('Adj Close', ticker)
                            close_col = ('Close', ticker)
                            volume_col = ('Volume', ticker)
                        else:
                            # Single-level columns
                            adj_close_col = 'Adj Close'
                            close_col = 'Close'
                            volume_col = 'Volume'
                        
                        # Check which columns are available and update
                        if adj_close_col in data.columns:
                            ticker_data.loc[common_dates, 'adj_close'] = data.loc[common_dates, adj_close_col]
                            logger.debug(f"Updated {ticker} with Adj Close data")
                        elif close_col in data.columns:
                            ticker_data.loc[common_dates, 'adj_close'] = data.loc[common_dates, close_col]
                            logger.debug(f"Updated {ticker} with Close data")
                        else:
                            logger.warning(f"No Close or Adj Close column found for {ticker}")
                        
                        if volume_col in data.columns:
                            ticker_data.loc[common_dates, 'volume'] = data.loc[common_dates, volume_col]
                        
                        # Mark as active during this period
                        active_mask = (ticker_data.index >= period_start) & (ticker_data.index <= period_end)
                        ticker_data.loc[active_mask, 'is_active'] = 1
                        got_data = True
                else:
                    logger.warning(f"No data downloaded for {ticker}")
                    
            except Exception as e:
                logger.warning(f"Error downloading {ticker}: {e}")
                continue
        
        if got_data:
            successful_downloads += 1
            logger.info(f"Successfully downloaded {ticker}")
        
        all_data[ticker] = ticker_data
    
    logger.info(f"Successfully downloaded data for {successful_downloads}/{len(tickers)} tickers")
    
    # Combine into multi-index DataFrame
    logger.info("Combining data into panel format...")
    panel = pd.concat(all_data, names=['ticker', 'date'])
    panel = panel.swaplevel().sort_index()
    
    # Save to file
    logger.info(f"Saving dataset to {output_file}")
    panel.to_parquet(output_file)
    
    # Print summary
    n_dates = len(panel.index.get_level_values('date').unique())
    n_tickers = len(panel.index.get_level_values('ticker').unique())
    logger.info(f"Dataset complete: {n_dates} dates x {n_tickers} tickers")
    
    return panel


def load_dataset(file_path='data/sp500_dataset.parquet'):
    """Load pre-built dataset from disk."""
    return pd.read_parquet(file_path)


def get_ticker_data(panel_df, ticker, start_date=None, end_date=None):
    """Extract data for a specific ticker."""
    ticker_data = panel_df.xs(ticker, level='ticker')
    if start_date:
        ticker_data = ticker_data[ticker_data.index >= start_date]
    if end_date:
        ticker_data = ticker_data[ticker_data.index <= end_date]
    return ticker_data


def get_active_tickers(panel_df, date):
    """Get all active tickers on a specific date."""
    if date in panel_df.index.get_level_values('date'):
        day_data = panel_df.loc[date]
        return day_data[day_data['is_active'] == 1].index.tolist()
    return []


def to_numpy_3d(panel_df):
    """Convert panel DataFrame to 3D NumPy array [dates x tickers x features]."""
    dates = panel_df.index.get_level_values('date').unique()
    tickers = panel_df.index.get_level_values('ticker').unique()
    features = panel_df.columns
    
    # Create 3D array
    arr = np.full((len(dates), len(tickers), len(features)), np.nan)
    
    for i, date in enumerate(dates):
        for j, ticker in enumerate(tickers):
            try:
                arr[i, j, :] = panel_df.loc[(date, ticker)].values
            except KeyError:
                pass  # Keep as NaN
    
    return arr, dates.values, tickers.values, features.tolist()


# Run the loader when script is executed directly
if __name__ == "__main__":
    print("Starting S&P 500 data download...")
    print("This may take a while depending on the number of tickers and date range...")
    
    # Run the loader
    panel_df = load_sp500_history(
        constituents_file='data/sp500_constituents.csv',
        start_date='1987-01-01',  # Adjust these dates as needed
        end_date='2025-01-01',
        output_file='data/sp500_dataset.parquet'
    )
    
    print("\nDataset built successfully!")
    print(f"Shape: {len(panel_df.index.get_level_values('date').unique())} dates x "
          f"{len(panel_df.index.get_level_values('ticker').unique())} tickers")
    
    # Show a sample of the data
    print("\nSample of data (first 5 rows):")
    print(panel_df.head())
    
    # Show some statistics
    active_count = panel_df['is_active'].sum()
    total_cells = len(panel_df)
    print(f"\nTotal data points: {total_cells:,}")
    print(f"Active data points: {active_count:,}")
    print(f"Coverage: {active_count/total_cells*100:.1f}%")