"""Data loader for variBAD trading - handles temporal splits and episode sampling."""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EpisodeSampler:
    """Randomly samples fixed-length episodes from data."""
    
    def __init__(self, data: pd.DataFrame, episode_length: int = 60):
        self.data = data
        self.episode_length = episode_length
        self.valid_start_indices = list(range(len(data) - episode_length + 1))
        
        if not self.valid_start_indices:
            raise ValueError(f"Data too short ({len(data)} days) for episodes of length {episode_length}")
    
    def sample_episode(self) -> pd.DataFrame:
        """Sample a random episode."""
        start_idx = np.random.choice(self.valid_start_indices)
        return self.data.iloc[start_idx:start_idx + self.episode_length].copy()
    
    def sample_batch(self, batch_size: int) -> List[pd.DataFrame]:
        """Sample multiple episodes."""
        return [self.sample_episode() for _ in range(batch_size)]


class DataLoader:
    """Fetch financial data and create train/test samplers."""
    
    def __init__(self, assets: List[str], start_date: str, end_date: str, 
                 train_end_date: str, max_retries: int = 3):
        """
        Initialize data loader with temporal split.
        
        Args:
            assets: List of ticker symbols
            start_date: Start of data period (YYYY-MM-DD)
            end_date: End of data period (YYYY-MM-DD)
            train_end_date: Cutoff date for train/test split (YYYY-MM-DD)
            max_retries: Maximum number of download attempts
        """
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date
        self.train_end_date = train_end_date
        self.max_retries = max_retries
        
        logger.info(f"Loading data for {assets} from {start_date} to {end_date}")
        logger.info(f"Train/test split at {train_end_date}")
        
        # Fetch and process data
        self.data = self._fetch_and_clean_data()
        
        # Create temporal split
        self._create_split()
        
    def _fetch_and_clean_data(self) -> pd.DataFrame:
        """Fetch data from yfinance with retries and clean it."""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Download attempt {attempt + 1}/{self.max_retries}")
                
                # Download data with explicit parameters to avoid warnings
                raw_data = yf.download(
                    self.assets,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=True,  # Explicitly set to avoid warning
                    threads=False  # Single thread to avoid timeout issues
                )
                
                # Check if we got data
                if raw_data.empty:
                    raise ValueError("No data downloaded")
                
                # Handle single asset case
                if len(self.assets) == 1:
                    raw_data.columns = pd.MultiIndex.from_product(
                        [raw_data.columns, self.assets]
                    )
                
                # Forward fill missing values using the new method
                data = raw_data.ffill().dropna()
                
                if len(data) == 0:
                    raise ValueError("No valid data after cleaning")
                
                logger.info(f"Successfully loaded {len(data)} days of data")
                logger.info(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
                
                return data
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 5  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error("All download attempts failed")
                    raise RuntimeError(f"Failed to download data after {self.max_retries} attempts: {e}")
    
    def _create_split(self):
        """Create train/test split based on cutoff date."""
        cutoff = pd.to_datetime(self.train_end_date)
        
        self.train_data = self.data[self.data.index <= cutoff]
        self.test_data = self.data[self.data.index > cutoff]
        
        if len(self.train_data) == 0:
            raise ValueError("No training data available")
        if len(self.test_data) == 0:
            raise ValueError("No test data available")
        
        logger.info(f"Train data: {len(self.train_data)} days "
                   f"({self.train_data.index[0].date()} to {self.train_data.index[-1].date()})")
        logger.info(f"Test data: {len(self.test_data)} days "
                   f"({self.test_data.index[0].date()} to {self.test_data.index[-1].date()})")
    
    def get_price_data(self, mode: str = 'train') -> Dict[str, pd.DataFrame]:
        """Get price data organized by type."""
        data = self.train_data if mode == 'train' else self.test_data
        
        return {
            'open': data['Open'],
            'high': data['High'],
            'low': data['Low'],
            'close': data['Close'],
            'volume': data['Volume']
        }
    
    def get_returns(self, mode: str = 'train') -> pd.DataFrame:
        """Calculate returns for train or test data."""
        data = self.train_data if mode == 'train' else self.test_data
        return data['Close'].pct_change().dropna()
    
    def get_episode_sampler(self, mode: str = 'train', 
                           episode_length: int = 60) -> EpisodeSampler:
        """Get episode sampler for train or test data."""
        data = self.train_data if mode == 'train' else self.test_data
        
        if len(data) < episode_length:
            raise ValueError(f"{mode} data too short: {len(data)} days < {episode_length} episode length")
            
        return EpisodeSampler(data, episode_length)
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of the data."""
        return {
            'assets': self.assets,
            'total_days': len(self.data),
            'train_days': len(self.train_data),
            'test_days': len(self.test_data),
            'date_range': f"{self.data.index[0].date()} to {self.data.index[-1].date()}",
            'train_range': f"{self.train_data.index[0].date()} to {self.train_data.index[-1].date()}",
            'test_range': f"{self.test_data.index[0].date()} to {self.test_data.index[-1].date()}"
        }


def test_data_loader():
    """Quick test of the data loader."""
    # Test with fewer assets first
    loader = DataLoader(
        assets=['SPY', 'QQQ'],  # Start with just 2 assets
        start_date='2018-01-01',
        end_date='2023-12-31',
        train_end_date='2021-12-31'
    )
    
    # Print summary
    summary = loader.get_data_summary()
    print("\nData Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test episode sampling
    train_sampler = loader.get_episode_sampler('train', episode_length=60)
    test_sampler = loader.get_episode_sampler('test', episode_length=60)
    
    # Sample a few episodes
    train_episodes = train_sampler.sample_batch(3)
    print(f"\nSampled {len(train_episodes)} training episodes")
    print(f"Episode shape: {train_episodes[0].shape}")
    
    return loader


if __name__ == "__main__":
    test_data_loader()