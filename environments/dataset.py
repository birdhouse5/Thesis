# In dataset.py
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

class Dataset:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.tickers = sorted(self.data['ticker'].unique())
        self.num_assets = len(self.tickers)
        self.dates = sorted(self.data['date'].unique())
        self.num_days = len(self.dates)
        
        # Select features for training
        self.feature_cols = self._select_training_features()
        self.num_features = len(self.feature_cols)
        print(f"Selected {self.num_features} features for training:")
        for i, col in enumerate(self.feature_cols):
            print(f"  {i:2d}: {col}")

    def _select_training_features(self):
        """Keep normalized features + unnormalized ones without normalized version"""
        all_cols = self.data.columns.tolist()

        # Get all normalized columns
        normalized_cols = [col for col in all_cols if col.endswith('_norm')]

        # Get unnormalized columns that don't have a normalized version
        unnormalized_cols = []
        for col in all_cols:
            if col in ['date', 'ticker']:
                continue
            if not col.endswith('_norm'):
                # Check if there's a normalized version
                norm_version = f"{col}_norm"
                if norm_version not in all_cols:
                    unnormalized_cols.append(col)

        # Combine and sort for consistency
        selected_cols = sorted(normalized_cols + unnormalized_cols)
        return selected_cols

    def get_window(self, start_day_idx, end_day_idx):
        """Return data shaped as (days, assets, features)"""
        window_dates = self.dates[start_day_idx:end_day_idx]
        window_data = self.data[self.data['date'].isin(window_dates)]

        # Sort by date then ticker to ensure consistent ordering
        window_data = window_data.sort_values(['date', 'ticker'])

        features = window_data[self.feature_cols].values
        
        # Reshape to (T, N, F)
        num_days = len(window_dates)
        return features.reshape(num_days, self.num_assets, self.num_features)
    
    def __len__(self):
        return self.num_days