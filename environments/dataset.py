import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple

class Dataset:
    """Simplified dataset with temporal splitting."""
    
    def __init__(self, data_path: str, split: str, train_end: str, val_end: str):
        """
        Initialize dataset with temporal split.
        
        Args:
            data_path: Path to parquet file
            split: 'train', 'val', or 'test'
            train_end: End date for training split (for SP500)
            val_end: End date for validation split (for SP500)
        """
        # Load full dataset
        full_data = pd.read_parquet(data_path)
        full_data['date'] = pd.to_datetime(full_data['date'])
        
        # Determine splitting method based on data characteristics
        date_range_days = (full_data['date'].max() - full_data['date'].min()).days
        
        if date_range_days > 10000:  # SP500 case (many years of daily data)
            self.data = self._date_based_split(full_data, split, train_end, val_end)
            self.split_info = f"{split} (date-based)"
        else:  # Crypto case (shorter time period, high frequency)
            self.data = self._proportional_split(full_data, split)
            self.split_info = f"{split} (proportional)"
        
        # Verify we have data
        if len(self.data) == 0:
            raise ValueError(f"No data found for {split} split")
        
        # Initialize properties
        self.split = split
        self.tickers = sorted(self.data['ticker'].unique())
        self.num_assets = len(self.tickers)
        self.dates = sorted(self.data['date'].unique())
        self.num_days = len(self.dates)
        
        # Feature selection
        self.feature_cols = [col for col in self.data.columns if col.endswith('_norm')]
        self.num_features = len(self.feature_cols)
        
        # Validation
        expected_rows = self.num_days * self.num_assets
        if len(self.data) != expected_rows:
            raise ValueError(f"{split} split not rectangular: {len(self.data)} != {expected_rows}")
        
        print(f"Dataset {self.split_info}:")
        print(f"  Dates: {self.data['date'].min().date()} to {self.data['date'].max().date()}")
        print(f"  Shape: {self.data.shape}")
        print(f"  Days: {self.num_days}, Assets: {self.num_assets}, Features: {self.num_features}")

    def _date_based_split(self, data: pd.DataFrame, split: str, train_end: str, val_end: str):
        """Split based on explicit dates (for SP500)."""
        train_end_date = pd.to_datetime(train_end)
        val_end_date = pd.to_datetime(val_end)
        
        if split == 'train':
            return data[data['date'] <= train_end_date].copy()
        elif split == 'val':
            return data[(data['date'] > train_end_date) & (data['date'] <= val_end_date)].copy()
        else:  # test
            return data[data['date'] > val_end_date].copy()

    def _proportional_split(self, data: pd.DataFrame, split: str, 
                          proportions: Tuple[float, float, float] = (0.7, 0.2, 0.1)):
        """Split based on proportions (for crypto)."""
        unique_dates = sorted(data['date'].unique())
        total_days = len(unique_dates)
        
        train_days = int(proportions[0] * total_days)
        val_days = int(proportions[1] * total_days)
        
        if split == 'train':
            split_dates = unique_dates[:train_days]
        elif split == 'val':
            split_dates = unique_dates[train_days:train_days + val_days]
        else:  # test
            split_dates = unique_dates[train_days + val_days:]
        
        return data[data['date'].isin(split_dates)].copy()

    def get_window(self, start_day_idx: int, end_day_idx: int) -> Dict[str, np.ndarray]:
        """Return features and prices for date range."""
        window_dates = self.dates[start_day_idx:end_day_idx]
        window_data = self.data[self.data['date'].isin(window_dates)]
        window_data = window_data.sort_values(['date', 'ticker'])
        
        # Reshape to (T, N, F) and (T, N)
        features = window_data[self.feature_cols].values
        features = features.reshape(len(window_dates), self.num_assets, self.num_features)
        
        prices = window_data['close'].values
        prices = prices.reshape(len(window_dates), self.num_assets)
        
        return {
            'features': features,
            'raw_prices': prices
        }

    def get_split_info(self) -> Dict:
        """Get split information."""
        return {
            'split': self.split,
            'split_info': self.split_info,
            'num_days': self.num_days,
            'num_assets': self.num_assets,
            'num_features': self.num_features,
            'date_range': (self.data['date'].min().date(), self.data['date'].max().date()),
            'total_samples': len(self.data)
        }

    def __len__(self):
        return self.num_days


def create_split_datasets(data_path: str, train_end: str = '2015-12-31', 
                         val_end: str = '2020-12-31') -> Dict[str, Dataset]:
    """Create train/val/test datasets with automatic split detection."""
    
    datasets = {}
    
    try:
        datasets['train'] = Dataset(data_path, 'train', train_end, val_end)
        datasets['val'] = Dataset(data_path, 'val', train_end, val_end)
        datasets['test'] = Dataset(data_path, 'test', train_end, val_end)
        
        print(f"\nSplit datasets created:")
        for name, dataset in datasets.items():
            info = dataset.get_split_info()
            print(f"  {name}: {info['num_days']} days, {info['date_range'][0]} to {info['date_range'][1]}")
        
        return datasets
        
    except Exception as e:
        print(f"Error creating datasets: {e}")
        raise