"""
Memory-efficient S&P 500 historical OHLCV data loader.
Processes data in batches to minimize RAM usage.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from pathlib import Path
import logging
import time
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MemoryEfficientOHLCVLoader:
    def __init__(self, constituents_file='sp500_constituents.csv', 
                 output_file='sp500_ohlcv_dataset.parquet',
                 batch_size=50, delay_between_requests=0.1):
        """
        Initialize the memory-efficient OHLCV loader.
        
        Args:
            constituents_file: Path to CSV with columns [ticker, start_date, end_date]
            output_file: Where to save the final dataset
            batch_size: Number of tickers to process in each batch
            delay_between_requests: Seconds to wait between API calls
        """
        self.constituents_file = constituents_file
        self.output_file = output_file
        self.batch_size = batch_size
        self.delay_between_requests = delay_between_requests
        
        # Create parent directory if needed
        Path(output_file).parent.mkdir(exist_ok=True)
    
    def load_sp500_history_efficient(self, start_date='1987-01-01', end_date='2025-01-01'):
        """
        Main function to load S&P 500 historical OHLCV data efficiently.
        """
        # Load constituents
        logger.info(f"Loading constituents from {self.constituents_file}")
        members = pd.read_csv(self.constituents_file)
        members['start_date'] = pd.to_datetime(members['start_date'])
        members['end_date'] = pd.to_datetime(members['end_date'])
        
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        global_start = pd.to_datetime(start_date)
        global_end = pd.to_datetime(end_date)
        
        # Get unique tickers
        tickers = members['ticker'].unique()
        logger.info(f"Found {len(tickers)} unique tickers")
        
        # Process in batches
        total_batches = (len(tickers) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(tickers))
            batch_tickers = tickers[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} "
                       f"(tickers {start_idx + 1}-{end_idx})")
            
            # Process batch
            batch_data = self._process_batch(
                batch_tickers, members, dates, global_start, global_end
            )
            
            # Save/update file
            self._save_batch_data(batch_data, batch_idx == 0)
            
            # Clear memory
            del batch_data
            
            logger.info(f"Batch {batch_idx + 1} completed and saved")
        
        # Load final dataset for summary
        final_df = pd.read_parquet(self.output_file)
        n_dates = len(final_df.index.get_level_values('date').unique())
        n_tickers = len(final_df.index.get_level_values('ticker').unique())
        logger.info(f"Dataset complete: {n_dates} dates x {n_tickers} tickers")
        
        return final_df
    
    def _process_batch(self, batch_tickers, members, dates, global_start, global_end):
        """Process a batch of tickers with enhanced missing value handling."""
        batch_data = {}
        successful_downloads = 0
        
        for i, ticker in enumerate(batch_tickers):
            if i % 10 == 0:
                logger.info(f"  Processing ticker {i + 1}/{len(batch_tickers)}: {ticker}")
            
            # Get membership periods for this ticker
            ticker_periods = members[members['ticker'] == ticker]
            
            # Initialize empty data with all OHLCV columns
            ticker_data = pd.DataFrame(index=dates)
            ticker_data['open'] = np.nan
            ticker_data['high'] = np.nan
            ticker_data['low'] = np.nan
            ticker_data['close'] = np.nan
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
                    period_end = period['end_date'] if pd.notna(period['end_date']) else pd.to_datetime('2025-01-01')
                    
                    # Download data with extended range to catch delisted stocks
                    download_end = min(period_end + pd.Timedelta(days=30), pd.to_datetime('2025-01-01'))
                    
                    data = yf.download(
                        ticker, 
                        start=period_start, 
                        end=download_end,
                        progress=False,
                        auto_adjust=False
                    )
                    
                    if not data.empty:
                        # Filter to only include dates within our global range
                        data = data[(data.index >= global_start) & 
                                   (data.index <= global_end)]
                        
                        if data.empty:
                            continue
                        
                        # Only update for dates that exist in both indices
                        common_dates = ticker_data.index.intersection(data.index)
                        
                        if len(common_dates) > 0:
                            # Handle multi-level columns from yfinance
                            if isinstance(data.columns, pd.MultiIndex):
                                open_col = ('Open', ticker)
                                high_col = ('High', ticker)
                                low_col = ('Low', ticker)
                                close_col = ('Close', ticker)
                                adj_close_col = ('Adj Close', ticker)
                                volume_col = ('Volume', ticker)
                            else:
                                open_col = 'Open'
                                high_col = 'High'
                                low_col = 'Low'
                                close_col = 'Close'
                                adj_close_col = 'Adj Close'
                                volume_col = 'Volume'
                            
                            # Update OHLCV data
                            if open_col in data.columns:
                                ticker_data.loc[common_dates, 'open'] = data.loc[common_dates, open_col]
                            
                            if high_col in data.columns:
                                ticker_data.loc[common_dates, 'high'] = data.loc[common_dates, high_col]
                            
                            if low_col in data.columns:
                                ticker_data.loc[common_dates, 'low'] = data.loc[common_dates, low_col]
                            
                            if close_col in data.columns:
                                ticker_data.loc[common_dates, 'close'] = data.loc[common_dates, close_col]
                            
                            if adj_close_col in data.columns:
                                ticker_data.loc[common_dates, 'adj_close'] = data.loc[common_dates, adj_close_col]
                            elif close_col in data.columns:
                                # Fallback to close if adj_close not available
                                ticker_data.loc[common_dates, 'adj_close'] = data.loc[common_dates, close_col]
                            
                            if volume_col in data.columns:
                                ticker_data.loc[common_dates, 'volume'] = data.loc[common_dates, volume_col]
                            
                            # Mark as active during this period
                            active_mask = (ticker_data.index >= period_start) & (ticker_data.index <= period_end)
                            ticker_data.loc[active_mask, 'is_active'] = 1
                            got_data = True
                        
                except Exception as e:
                    logger.warning(f"Error downloading {ticker}: {e}")
                    continue
                
                # Rate limiting
                time.sleep(self.delay_between_requests)
            
            if got_data:
                # Apply missing value handling
                ticker_data = self._apply_missing_value_handling(ticker_data, ticker)
                successful_downloads += 1
                batch_data[ticker] = ticker_data
            
        logger.info(f"  Successfully downloaded {successful_downloads}/{len(batch_tickers)} tickers in batch")
        
        # Convert to multi-index DataFrame
        if batch_data:
            panel = pd.concat(batch_data, names=['ticker', 'date'])
            panel = panel.swaplevel().sort_index()
            return panel
        else:
            return pd.DataFrame()
    
    def _apply_missing_value_handling(self, ticker_data, ticker):
        """Apply missing value handling strategies for OHLCV data."""
        # Only process active periods
        active_mask = ticker_data['is_active'] == 1
        
        if not active_mask.any():
            return ticker_data
        
        # Forward fill for short gaps (1-5 business days) in active periods
        # Get active period boundaries
        active_data = ticker_data[active_mask].copy()
        
        # Forward fill prices for gaps <= 5 days
        price_columns = ['open', 'high', 'low', 'close', 'adj_close']
        
        for col in price_columns:
            price_series = active_data[col]
            filled_series = price_series.copy()
            
            # Identify gaps and fill short ones
            is_na = price_series.isna()
            if is_na.any():
                # Find consecutive NaN groups
                na_groups = (is_na != is_na.shift()).cumsum()[is_na]
                
                for group_id in na_groups.unique():
                    group_mask = na_groups == group_id
                    gap_length = group_mask.sum()
                    
                    if gap_length <= 5:  # Fill gaps of 5 days or less
                        gap_indices = na_groups[group_mask].index
                        
                        # Find the last valid price before the gap
                        start_idx = gap_indices[0]
                        try:
                            last_valid_idx = price_series.loc[:start_idx].last_valid_index()
                            if last_valid_idx is not None:
                                last_valid_price = price_series.loc[last_valid_idx]
                                filled_series.loc[gap_indices] = last_valid_price
                        except:
                            continue
            
            # Update the active data with filled prices
            ticker_data.loc[active_mask, col] = filled_series
        
        # Handle volume: set to 0 for non-trading days in active periods
        volume_series = ticker_data.loc[active_mask, 'volume']
        
        # For missing volume in active periods, set to 0 (assumption: no trading occurred)
        ticker_data.loc[active_mask & ticker_data['volume'].isna(), 'volume'] = 0
        
        return ticker_data
    
    def _save_batch_data(self, batch_data, is_first_batch):
        """Save batch data to file, either creating new file or appending."""
        if batch_data.empty:
            logger.warning("No data to save for this batch")
            return
        
        if is_first_batch:
            # Create new file
            batch_data.to_parquet(self.output_file)
            logger.info(f"Created new file: {self.output_file}")
        else:
            # Append to existing file
            if os.path.exists(self.output_file):
                existing_data = pd.read_parquet(self.output_file)
                combined_data = pd.concat([existing_data, batch_data])
                combined_data = combined_data.sort_index()
                combined_data.to_parquet(self.output_file)
                logger.info(f"Appended batch to existing file: {self.output_file}")
            else:
                # Fallback if file doesn't exist
                batch_data.to_parquet(self.output_file)
                logger.info(f"Created new file: {self.output_file}")
    
    def resume_from_checkpoint(self, start_date='1987-01-01', end_date='2025-01-01'):
        """Resume download from where it left off."""
        if not os.path.exists(self.output_file):
            logger.info("No existing file found, starting fresh")
            return self.load_sp500_history_efficient(start_date, end_date)
        
        # Load existing data to see what we have
        existing_data = pd.read_parquet(self.output_file)
        existing_tickers = set(existing_data.index.get_level_values('ticker').unique())
        
        # Load constituents to see what we need
        members = pd.read_csv(self.constituents_file)
        all_tickers = set(members['ticker'].unique())
        
        # Find missing tickers
        missing_tickers = all_tickers - existing_tickers
        
        if not missing_tickers:
            logger.info("All tickers already processed!")
            return existing_data
        
        logger.info(f"Found {len(missing_tickers)} missing tickers to process")
        
        # Process missing tickers using the same batch logic
        missing_list = list(missing_tickers)
        members_filtered = members[members['ticker'].isin(missing_list)]
        
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        global_start = pd.to_datetime(start_date)
        global_end = pd.to_datetime(end_date)
        
        # Process in batches
        total_batches = (len(missing_list) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(missing_list))
            batch_tickers = missing_list[start_idx:end_idx]
            
            logger.info(f"Processing missing batch {batch_idx + 1}/{total_batches}")
            
            # Process batch
            batch_data = self._process_batch(
                batch_tickers, members_filtered, dates, global_start, global_end
            )
            
            # Append to existing file
            self._save_batch_data(batch_data, is_first_batch=False)
            
            # Clear memory
            del batch_data
        
        # Return final dataset
        return pd.read_parquet(self.output_file)


# Convenience functions
def load_sp500_ohlcv_efficient(
    constituents_file='data/sp500_constituents.csv',
    start_date='1987-01-01',
    end_date='2025-01-01',
    output_file='data/sp500_ohlcv_dataset.parquet',
    batch_size=50,
    delay_between_requests=0.1
):
    """
    Load S&P 500 OHLCV data efficiently with memory management.
    """
    loader = MemoryEfficientOHLCVLoader(
        constituents_file=constituents_file,
        output_file=output_file,
        batch_size=batch_size,
        delay_between_requests=delay_between_requests
    )
    
    return loader.load_sp500_history_efficient(start_date, end_date)


def resume_ohlcv_download(
    constituents_file='data/sp500_constituents.csv',
    start_date='1987-01-01',
    end_date='2025-01-01',
    output_file='data/sp500_ohlcv_dataset.parquet',
    batch_size=50,
    delay_between_requests=0.1
):
    """
    Resume interrupted OHLCV download.
    """
    loader = MemoryEfficientOHLCVLoader(
        constituents_file=constituents_file,
        output_file=output_file,
        batch_size=batch_size,
        delay_between_requests=delay_between_requests
    )
    
    return loader.resume_from_checkpoint(start_date, end_date)


# Keep the utility functions from the original
def load_ohlcv_dataset(file_path='data/sp500_ohlcv_dataset.parquet'):
    """Load pre-built OHLCV dataset from disk."""
    return pd.read_parquet(file_path)


def get_ohlcv_ticker_data(panel_df, ticker, start_date=None, end_date=None):
    """Extract OHLCV data for a specific ticker."""
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


# Run the loader when script is executed directly
if __name__ == "__main__":
    print("Starting memory-efficient S&P 500 OHLCV data download...")
    print("Processing in batches to minimize memory usage...")
    
    # Initialize loader
    loader = MemoryEfficientOHLCVLoader(
        constituents_file='data/sp500_constituents.csv',
        output_file='data/sp500_ohlcv_dataset.parquet',
        batch_size=50,
        delay_between_requests=0.1
    )
    
    # Run the efficient loader
    panel_df = loader.load_sp500_history_efficient(
        start_date='1987-01-01',
        end_date='2025-01-01'
    )
    
    print("\nOHLCV Dataset built successfully!")
    print(f"Shape: {len(panel_df.index.get_level_values('date').unique())} dates x "
          f"{len(panel_df.index.get_level_values('ticker').unique())} tickers")
    
    # Show a sample of the data
    print("\nSample of OHLCV data (first 5 rows):")
    print(panel_df.head())
    
    # Show column structure
    print(f"\nColumns: {list(panel_df.columns)}")