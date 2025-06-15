"""Data loader for fetching financial data."""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Fetch and validate financial data from yfinance."""
    
    def __init__(self, assets: List[str], start_date: str, end_date: str):
        """
        Initialize data loader.
        
        Args:
            assets: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch data from yfinance."""
        logger.info(f"Fetching data for {self.assets} from {self.start_date} to {self.end_date}")
        
        # Add buffer for technical indicators
        buffer_start = (pd.to_datetime(self.start_date) - timedelta(days=100)).strftime('%Y-%m-%d')
        
        # Download data
        self.data = yf.download(
            self.assets,
            start=buffer_start,
            end=self.end_date,
            progress=False
        )
        
        # Handle single asset case
        if len(self.assets) == 1:
            # Add asset level to columns
            self.data.columns = pd.MultiIndex.from_product(
                [self.data.columns, self.assets]
            )
        
        logger.info(f"Fetched data shape: {self.data.shape}")
        logger.info(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
        
        return self.data
    
    def validate_data(self, min_common_dates: float = 0.95) -> Dict:
        """Validate data quality."""
        if self.data is None:
            raise ValueError("No data loaded. Call fetch_data first.")
        
        validation_report = {
            'total_days': len(self.data),
            'assets': {},
            'issues': []
        }
        
        for asset in self.assets:
            # Count non-null values for each asset
            asset_data = self.data['Close'][asset] if len(self.assets) > 1 else self.data['Close']
            non_null = asset_data.notna().sum()
            null_count = asset_data.isna().sum()
            
            validation_report['assets'][asset] = {
                'non_null_days': non_null,
                'null_days': null_count,
                'coverage': non_null / len(self.data)
            }
            
            if validation_report['assets'][asset]['coverage'] < min_common_dates:
                validation_report['issues'].append(
                    f"{asset} has only {validation_report['assets'][asset]['coverage']:.1%} coverage"
                )
        
        # Check for common trading days
        if len(self.assets) > 1:
            close_data = self.data['Close']
            common_days = close_data.notna().all(axis=1).sum()
            validation_report['common_trading_days'] = common_days
            validation_report['common_coverage'] = common_days / len(self.data)
            
            if validation_report['common_coverage'] < min_common_dates:
                validation_report['issues'].append(
                    f"Only {validation_report['common_coverage']:.1%} common trading days"
                )
        
        return validation_report
    
    def clean_data(self, method: str = 'forward_fill') -> pd.DataFrame:
        """Clean data by handling missing values."""
        if self.data is None:
            raise ValueError("No data loaded. Call fetch_data first.")
        
        logger.info(f"Cleaning data using method: {method}")
        
        if method == 'forward_fill':
            self.data = self.data.fillna(method='ffill')
        elif method == 'drop':
            self.data = self.data.dropna()
        elif method == 'interpolate':
            self.data = self.data.interpolate(method='linear')
        else:
            raise ValueError(f"Unknown cleaning method: {method}")
        
        # Remove any remaining NaN at the beginning
        self.data = self.data.dropna()
        
        logger.info(f"Cleaned data shape: {self.data.shape}")
        
        return self.data
    
    def get_price_data(self) -> Dict[str, pd.DataFrame]:
        """Get price data organized by type."""
        if self.data is None:
            raise ValueError("No data loaded. Call fetch_data first.")
        
        price_data = {
            'open': self.data['Open'],
            'high': self.data['High'],
            'low': self.data['Low'],
            'close': self.data['Close'],
            'volume': self.data['Volume']
        }
        
        return price_data


def quick_test():
    """Quick test of data loader."""
    # Test configuration
    assets = ['SPY', 'QQQ', 'TLT']
    start_date = '2020-01-01'
    end_date = '2021-12-31'
    
    # Create loader
    loader = DataLoader(assets, start_date, end_date)
    
    # Fetch data
    data = loader.fetch_data()
    print(f"\nFetched data shape: {data.shape}")
    
    # Validate
    report = loader.validate_data()
    print(f"\nValidation report:")
    print(f"Total days: {report['total_days']}")
    print(f"Common coverage: {report.get('common_coverage', 1):.1%}")
    
    if report['issues']:
        print("Issues found:")
        for issue in report['issues']:
            print(f"  - {issue}")
    else:
        print("No issues found!")
    
    # Clean
    clean_data = loader.clean_data()
    print(f"\nCleaned data shape: {clean_data.shape}")
    
    return loader


if __name__ == "__main__":
    quick_test()