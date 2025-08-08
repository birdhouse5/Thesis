# In dataset.py
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

class Dataset:
    def __init__(self, data_path):
        self.data = pd.read_parquet(data_path)
        self.tickers = sorted(self.data['ticker'].unique())
        self.num_assets = len(self.tickers)
        self.dates = sorted(self.data['date'].unique())
        self.num_days = len(self.dates)
        
        # Select features for training
        self.feature_cols = self._select_training_features()
        self.num_features = len(self.feature_cols)

        expected_rows = self.num_days * self.num_assets
        actual_rows = len(self.data)
        if actual_rows != expected_rows:
            raise ValueError(f"Data not rectangular: {actual_rows} rows, expected {expected_rows}")
        
        print(f"Selected {self.num_features} features for training:")
        for i, col in enumerate(self.feature_cols):
            print(f"  {i:2d}: {col}")

    def _select_training_features(self):
        """Use only normalized features for consistent scaling"""
        normalized_cols = [col for col in self.data.columns if col.endswith('_norm')]
        return sorted(normalized_cols)

    def get_window(self, start_day_idx, end_day_idx):
        """Return normalized features and raw prices"""
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
    
    def __len__(self):
        return self.num_days