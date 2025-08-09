import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple

class Dataset:
    def __init__(self, data_path, split='train', train_end='2015-12-31', val_end='2020-12-31'):
        """
        Initialize dataset with temporal split support.
        
        Args:
            data_path: Path to the parquet file
            split: 'train', 'val', or 'test'
            train_end: End date for training split (inclusive)
            val_end: End date for validation split (inclusive)
        """
        # Load full dataset
        full_data = pd.read_parquet(data_path)
        full_data['date'] = pd.to_datetime(full_data['date'])
        
        # Define split boundaries
        train_end_date = pd.to_datetime(train_end)
        val_end_date = pd.to_datetime(val_end)

        # Convert split dates to match the data's timezone if data has timezone
        if full_data['date'].dt.tz is not None:
            if train_end_date.tz is None:
                train_end_date = train_end_date.tz_localize(full_data['date'].dt.tz)
            else:
                train_end_date = train_end_date.tz_convert(full_data['date'].dt.tz)
                
            if val_end_date.tz is None:
                val_end_date = val_end_date.tz_localize(full_data['date'].dt.tz)
            else:
                val_end_date = val_end_date.tz_convert(full_data['date'].dt.tz)


        # Apply temporal split
        if split == 'train':
            self.data = full_data[full_data['date'] <= train_end_date].copy()
            self.split_name = f"train (up to {train_end})"
        elif split == 'val':
            self.data = full_data[
                (full_data['date'] > train_end_date) & 
                (full_data['date'] <= val_end_date)
            ].copy()
            self.split_name = f"val ({train_end} to {val_end})"
        elif split == 'test':
            self.data = full_data[full_data['date'] > val_end_date].copy()
            self.split_name = f"test (after {val_end})"
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        # Verify we have data
        if len(self.data) == 0:
            raise ValueError(f"No data found for {split} split")
        
        # Initialize dataset properties
        self.split = split
        self.tickers = sorted(self.data['ticker'].unique())
        self.num_assets = len(self.tickers)
        self.dates = sorted(self.data['date'].unique())
        self.num_days = len(self.dates)
        
        # Select features for training
        self.feature_cols = self._select_training_features()
        self.num_features = len(self.feature_cols)

        # Verify rectangular structure
        expected_rows = self.num_days * self.num_assets
        actual_rows = len(self.data)
        if actual_rows != expected_rows:
            print(f"Warning: {split} split not perfectly rectangular: {actual_rows} rows, expected {expected_rows}")
            # Clean up any missing combinations
            self._ensure_rectangular()
        
        print(f"Dataset {self.split_name}:")
        print(f"  Dates: {self.data['date'].min().date()} to {self.data['date'].max().date()}")
        print(f"  Shape: {self.data.shape}")
        print(f"  Days: {self.num_days}, Assets: {self.num_assets}, Features: {self.num_features}")

    def _select_training_features(self):
        """Use only normalized features for consistent scaling"""
        normalized_cols = [col for col in self.data.columns if col.endswith('_norm')]
        return sorted(normalized_cols)
    
    def _ensure_rectangular(self):
        """Ensure all date-ticker combinations exist"""
        # Create complete index
        complete_idx = pd.MultiIndex.from_product(
            [self.dates, self.tickers],
            names=['date', 'ticker']
        )
        
        # Reindex and fill missing values
        self.data = self.data.set_index(['date', 'ticker']).reindex(complete_idx).reset_index()
        
        # Forward fill missing values within each ticker
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data.groupby('ticker')[numeric_cols].ffill().bfill()
        
        # Update counts
        self.dates = sorted(self.data['date'].unique())
        self.num_days = len(self.dates)

    def get_window(self, start_day_idx, end_day_idx):
        """Return normalized features and raw prices for date range"""
        window_dates = self.dates[start_day_idx:end_day_idx]
        window_data = self.data[self.data['date'].isin(window_dates)]
        window_data = window_data.sort_values(['date', 'ticker'])
        
        # Normalized features for VAE/policy
        features = window_data[self.feature_cols].values
        features = features.reshape(len(window_dates), self.num_assets, self.num_features)
        
        # Raw prices for return calculations  
        raw_prices = window_data['close'].values
        raw_prices = raw_prices.reshape(len(window_dates), self.num_assets)
        
        return {
            'features': features,      # (T, N, F)
            'raw_prices': raw_prices   # (T, N)
        }   
    
    def get_split_info(self):
        """Get information about the current split"""
        return {
            'split': self.split,
            'split_name': self.split_name,
            'num_days': self.num_days,
            'num_assets': self.num_assets,
            'num_features': self.num_features,
            'date_range': (self.data['date'].min().date(), self.data['date'].max().date()),
            'total_samples': len(self.data)
        }
    
    def __len__(self):
        return self.num_days


def create_split_datasets(data_path, train_end='2015-12-31', val_end='2020-12-31'):
    """
    Create train, validation, and test datasets with temporal split.
    
    Args:
        data_path: Path to the full dataset
        train_end: End date for training (inclusive)
        val_end: End date for validation (inclusive)
    
    Returns:
        Dictionary with train, val, test Dataset objects
    """
    datasets = {}
    
    try:
        datasets['train'] = Dataset(data_path, 'train', train_end, val_end)
        datasets['val'] = Dataset(data_path, 'val', train_end, val_end)
        datasets['test'] = Dataset(data_path, 'test', train_end, val_end)
        
        print(f"\n✅ Successfully created all splits:")
        for split_name, dataset in datasets.items():
            info = dataset.get_split_info()
            print(f"  {split_name}: {info['num_days']} days, {info['date_range'][0]} to {info['date_range'][1]}")
        
        return datasets
        
    except Exception as e:
        print(f"❌ Error creating splits: {e}")
        raise