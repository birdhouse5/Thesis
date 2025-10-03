"""
Unified Data Module for VariBAD Portfolio Optimization
Merged from data_preparation.py and dataset.py - single source of truth
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Iterator
from datetime import datetime, timedelta
import random
import requests
import torch
import pickle

logger = logging.getLogger(__name__)

# Constants from original data_preparation.py
SP500_TICKERS = [
    'IBM', 'MSFT', 'ORCL', 'INTC', 'HPQ', 'CSCO',  # Tech
    'JPM', 'BAC', 'WFC', 'C', 'AXP',              # Financial
    'JNJ', 'PFE', 'MRK', 'ABT',                   # Healthcare
    'KO', 'PG', 'WMT', 'PEP',                     # Consumer Staples
    'XOM', 'CVX', 'COP',                          # Energy
    'GE', 'CAT', 'BA',                            # Industrials
    'HD', 'MCD',                                  # Consumer Disc
    'SO', 'D',                                    # Utilities
    'DD'                                          # Materials
]

# Coverage window discovered by scanner
CRYPTO_START_DATE = datetime(2019, 4, 18, 6, 0, 0)
CRYPTO_END_DATE = datetime(2025, 10, 3, 13, 45, 0)
CRYPTO_TICKERS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "NEOUSDT", "LTCUSDT",
    "QTUMUSDT", "ADAUSDT", "XRPUSDT", "IOTAUSDT", "TUSDUSDT",
    "XLMUSDT", "ONTUSDT", "TRXUSDT", "ETCUSDT", "ICXUSDT",
    "VETUSDT", "USDCUSDT", "LINKUSDT", "ONGUSDT", "HOTUSDT",
    "ZILUSDT", "FETUSDT", "ZRXUSDT", "BATUSDT", "ZECUSDT",
    "IOSTUSDT", "CELRUSDT", "DASHUSDT", "THETAUSDT", "ENJUSDT"
]



# ETF_TICKERS = [
    
#     "SPY","QQQ","IWM","DIA","VTV","VUG",                            # US Market Style
#     "EFA","EEM","EWJ","EWU","EWG","EWY","INDA","MCHI",              # Global Equities
#     "TLT","IEF","SHY","AGG","LQD","HYG",                            # Bonds
#     "GLD","SLV","DBC","USO","UNG"                                   # Commodities
# ]


BASE_URL = "https://api.binance.com/api/v3/klines"


class DatasetSplit:
    """Represents a single temporal split (train/val/test) with RL-ready functionality"""
    
    def __init__(self, data: pd.DataFrame, split_name: str, feature_cols: List[str]):
        self.data = data
        self.split_name = split_name
        self.feature_cols = feature_cols
        
        # Dataset properties
        self.tickers = sorted(data['ticker'].unique())
        self.num_assets = len(self.tickers)
        self.timestamps = sorted(data['date'].unique())
        self.num_days = len(self.timestamps)
        self.dates = self.timestamps  # alias
        self.num_features = len(feature_cols)
        
        logger.info(f"Dataset split '{split_name}': {len(data)} rows, {self.num_days} days, {self.num_assets} assets, {self.num_features} features")
    
    def get_window(self, start_day_idx: int, end_day_idx: int) -> Dict[str, np.ndarray]:
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
    
    def get_window_tensor(self, start_day_idx: int, end_day_idx: int, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Direct tensor output - eliminates intermediate conversions"""
        window = self.get_window(start_day_idx, end_day_idx)
        return {
            'features': torch.tensor(window['features'], dtype=torch.float32, device=device),
            'raw_prices': torch.tensor(window['raw_prices'], dtype=torch.float32, device=device)
        }
    
    def get_split_info(self) -> Dict:
        """Get information about the current split"""
        return {
            'split': self.split_name,
            'num_days': self.num_days,
            'num_assets': self.num_assets,
            'num_features': self.num_features,
            'date_range': (self.data['date'].min().date(), self.data['date'].max().date()),
            'total_samples': len(self.data)
        }
    
    def __len__(self):
        return self.num_days


class PortfolioDataset:
    """
    Unified dataset class handling raw data → RL training
    Combines functionality from data_preparation.py and dataset.py
    """
    
    def __init__(self, asset_class: str, data_path: Optional[str] = None, 
                 force_recreate: bool = False, split: Optional[str] = None,
                 train_end: str = '2015-12-31', val_end: str = '2020-12-31',
                 proportional: bool = False, proportions: Tuple[float, float, float] = (0.7, 0.2, 0.1)):
        """
        Unified dataset class handling raw data → RL training
        
        Args:
            asset_class: "sp500" or "crypto"
            data_path: Optional path override
            force_recreate: Force regeneration of data
            split: Optional specific split to load ("train"/"val"/"test"/None for all)
            train_end: End date for training split
            val_end: End date for validation split
            proportional: Use proportional splitting instead of date-based
            proportions: (train, val, test) fractions for proportional mode
        """
        self.asset_class = asset_class
        self.data_path = data_path or f"environments/data/{asset_class}_rl_ready_cleaned.parquet"
        self.train_end = train_end
        self.val_end = val_end
        self.proportional = proportional
        self.proportions = proportions
        
        # Core data - will be populated by _load_or_create_data
        self.full_data = None
        self.splits = {}  # Will contain train/val/test DatasetSplit objects
        self.feature_cols = None
        
        # Load or create the core dataset
        self._load_or_create_data(force_recreate)
        
        # Set up temporal splits
        self._setup_splits()
        
        # If specific split requested, set convenience properties for backward compatibility
        if split:
            self._setup_single_split(split)
    
    def _load_or_create_data(self, force_recreate: bool):
        """Load existing data or create from scratch"""
        data_path = Path(self.data_path)
        
        if data_path.exists() and not force_recreate:
            logger.info(f"Loading existing dataset: {data_path}")
            self.full_data = pd.read_parquet(data_path)
        else:
            logger.info(f"Creating {self.asset_class} dataset from scratch...")
            data_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.asset_class == "sp500":
                self.full_data = self._create_sp500_dataset()
            elif self.asset_class == "crypto":
                self.full_data = self._create_crypto_dataset()
            else:
                raise ValueError(f"Unknown asset class: {self.asset_class}")
            
            # Save for future use
            self.full_data.to_parquet(data_path)
            logger.info(f"Dataset saved to {data_path}")
        
        # Set feature columns
        self.feature_cols = self._select_training_features()
        logger.info(f"Loaded dataset: {self.full_data.shape}, features: {len(self.feature_cols)}")
    
    def _create_sp500_dataset(self) -> pd.DataFrame:
        """Create SP500 dataset - migrated from data_preparation.py"""
        # Step 1: Download raw data
        raw_data = self._download_stock_data(SP500_TICKERS, '1990-02-16', '2025-01-01')
        
        # Step 2: Add technical indicators
        with_indicators = self._add_technical_indicators(raw_data)
        
        # Step 3: Normalize features
        normalized = self._normalize_features(with_indicators)
        
        # Step 4: Clean data
        cleaned = self._clean_data(normalized)
        
        return cleaned
    
    def _create_crypto_dataset(self) -> pd.DataFrame:
        """Create crypto dataset using chunked download"""
        # Use chunked download for full coverage
        raw_data = self._download_crypto_chunked(
            symbols=CRYPTO_TICKERS,
            start_date=CRYPTO_START_DATE,
            end_date=CRYPTO_END_DATE,
            interval="15m",
            chunk_days=30
        )
        
        # Process same as stocks
        with_indicators = self._add_technical_indicators(raw_data)
        normalized = self._normalize_features(with_indicators)
        cleaned = self._clean_data(normalized)
        
        return cleaned
    
    def _setup_splits(self):
        """Set up train/val/test splits"""
        self.full_data['date'] = pd.to_datetime(self.full_data['date'])
        
        if self.proportional:
            # Proportional splitting
            unique_dates = sorted(self.full_data['date'].unique())
            num_days = len(unique_dates)
            
            if abs(sum(self.proportions) - 1.0) > 1e-6:
                raise ValueError(f"Proportions must sum to 1. Got {self.proportions}")
            
            train_days = int(self.proportions[0] * num_days)
            val_days = int(self.proportions[1] * num_days)
            
            train_end_date = unique_dates[train_days - 1]
            val_end_date = unique_dates[train_days + val_days - 1]
            
            # Update for consistency
            self.train_end = train_end_date.strftime("%Y-%m-%d")
            self.val_end = val_end_date.strftime("%Y-%m-%d")
        
        # Date-based splitting
        train_end_date = pd.to_datetime(self.train_end)
        val_end_date = pd.to_datetime(self.val_end)
        
        # Handle timezone matching
        if self.full_data['date'].dt.tz is not None:
            if train_end_date.tz is None:
                train_end_date = train_end_date.tz_localize(self.full_data['date'].dt.tz)
            if val_end_date.tz is None:
                val_end_date = val_end_date.tz_localize(self.full_data['date'].dt.tz)
        
        # Create splits
        train_data = self.full_data[self.full_data['date'] <= train_end_date].copy()
        val_data = self.full_data[
            (self.full_data['date'] > train_end_date) & 
            (self.full_data['date'] <= val_end_date)
        ].copy()
        test_data = self.full_data[self.full_data['date'] > val_end_date].copy()
        
        # Verify we have data for all splits
        for split_name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            if len(data) == 0:
                raise ValueError(f"No data found for {split_name} split")
        
        # Create DatasetSplit objects
        self.splits['train'] = DatasetSplit(train_data, 'train', self.feature_cols)
        self.splits['val'] = DatasetSplit(val_data, 'val', self.feature_cols)
        self.splits['test'] = DatasetSplit(test_data, 'test', self.feature_cols)
        
        mode_info = f"Proportional splits" if self.proportional else f"Date-based splits (train_end={self.train_end}, val_end={self.val_end})"
        logger.info(f"Created splits using {mode_info}")
    
    def _setup_single_split(self, split: str):
        """Set up backward compatibility properties for single split usage"""
        if split not in self.splits:
            raise ValueError(f"Invalid split: {split}")
        
        # Mirror old Dataset class properties
        split_obj = self.splits[split]
        self.data = split_obj.data
        self.split = split
        self.split_name = f"{split} ({split_obj.get_split_info()['date_range']})"
        self.tickers = split_obj.tickers
        self.num_assets = split_obj.num_assets
        self.timestamps = split_obj.timestamps
        self.num_intervals = split_obj.num_days
        self.num_days = split_obj.num_days
        self.dates = split_obj.dates
        self.num_features = split_obj.num_features
    
    def get_split(self, split: str) -> DatasetSplit:
        """Get specific split"""
        return self.splits[split]
    
    def get_all_splits(self) -> Dict[str, DatasetSplit]:
        """Get all splits - replaces create_split_datasets"""
        return self.splits
    
    def _select_training_features(self) -> List[str]:
        """Use only normalized features for consistent scaling"""
        normalized_cols = [col for col in self.full_data.columns if col.endswith('_norm')]
        return sorted(normalized_cols)
    
    # ========== MIGRATED FUNCTIONS FROM data_preparation.py ==========

    # def _download_stock_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    #     """Download stock data using yfinance (safe version with yf.download)"""
    #     logger.info(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}")

    #     all_data = []
    #     failed_tickers = []

    #     for i, ticker in enumerate(tickers):
    #         try:
    #             logger.info(f"  • {ticker} ({i+1}/{len(tickers)})")

    #             # Use yf.download instead of Ticker().history to avoid curl/TLS issues
    #             hist = yf.download(
    #                 ticker,
    #                 start=start_date,
    #                 end=end_date,
    #                 auto_adjust=True,
    #                 progress=False,
    #                 threads=False  # critical: avoids curl backend
    #             )

    #             if hist.empty:
    #                 logger.warning(f"      No data found for {ticker}")
    #                 failed_tickers.append(ticker)
    #                 continue

    #             # Reset index and add ticker column
    #             hist = hist.reset_index()
    #             hist['ticker'] = ticker

    #             # Standardize column names
    #             hist = hist.rename(columns={
    #                 'Date': 'date',
    #                 'Open': 'open',
    #                 'High': 'high',
    #                 'Low': 'low',
    #                 'Close': 'close',
    #                 'Volume': 'volume'
    #             })

    #             hist['adj_close'] = hist['close']  # already adjusted by yfinance

    #             # Keep only relevant columns
    #             hist = hist[['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]
    #             all_data.append(hist)

    #         except Exception as e:
    #             logger.error(f"     Failed to download {ticker}: {e}")
    #             failed_tickers.append(ticker)
    #             continue

    #     if not all_data:
    #         raise ValueError("No data was successfully downloaded")

    #     # Combine and sort all tickers’ data
    #     combined_data = pd.concat(all_data, ignore_index=True)
    #     combined_data = combined_data.sort_values(['date', 'ticker']).reset_index(drop=True)

    #     logger.info(f"Downloaded data: {combined_data.shape}")
    #     if failed_tickers:
    #         logger.warning(f"Failed tickers: {', '.join(failed_tickers)}")

    #     return combined_data


    def _download_stock_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Download stock data using yfinance - migrated from data_preparation.py"""
        logger.info(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        all_data = []
        failed_tickers = []
        
        for i, ticker in enumerate(tickers):
            try:
                logger.info(f"  • {ticker} ({i+1}/{len(tickers)})")
                
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date, auto_adjust=True)
                
                if hist.empty:
                    logger.warning(f"      No data found for {ticker}")
                    failed_tickers.append(ticker)
                    continue
                
                # Reset index and add ticker
                hist = hist.reset_index()
                hist['ticker'] = ticker
                
                # Standardize column names
                hist = hist.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high', 
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                hist['adj_close'] = hist['close']  # Already adjusted by yfinance
                
                # Select relevant columns
                hist = hist[['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]
                all_data.append(hist)
                
            except Exception as e:
                logger.error(f"     Failed to download {ticker}: {e}")
                failed_tickers.append(ticker)
                continue
        
        if not all_data:
            raise ValueError("No data was successfully downloaded")
        
        # Combine and sort
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.sort_values(['date', 'ticker']).reset_index(drop=True)
        
        logger.info(f"Downloaded data: {combined_data.shape}")
        if failed_tickers:
            logger.warning(f"Failed tickers: {', '.join(failed_tickers)}")
        
        return combined_data
    
    # def _sample_crypto(self, symbols: List[str], attempts: int = 3, days: int = 92, 
    #                   interval: str = "15m", target_rows: int = 263520) -> pd.DataFrame:
    #     """Sample crypto OHLCV data - migrated from data_preparation.py"""
    #     for attempt in range(attempts):
    #         logger.info(f"Crypto sampling attempt {attempt+1} ({interval})")
    #         end = datetime.utcnow()
    #         start_bound = datetime(2024, 4, 2)
    #         max_start = end - timedelta(days=days)
    #         if start_bound >= max_start:
    #             start = start_bound
    #         else:
    #             start = start_bound + (max_start - start_bound) * random.random()
    #         end = start + timedelta(days=days)

    #         all_dfs, failed, illiquid = [], [], []
    #         for sym in symbols:
    #             df = self._fetch_klines(sym, interval, start, end)
    #             expected_rows = days * (1440 // 15)  # 96 per day
    #             if df is None or len(df) < expected_rows:
    #                 failed.append(sym)
    #                 continue
    #             if df["volume"].sum() <= 0:
    #                 illiquid.append(sym)
    #                 continue
    #             df["ticker"] = sym
    #             all_dfs.append(df)
            
    #         if not failed and not illiquid:
    #             full = pd.concat(all_dfs, ignore_index=True)
    #             logger.info(f"✅ Crypto sampling success: {full.shape}")
    #             return full.iloc[:target_rows]
    #         else:
    #             logger.info(f"Retrying due to failed/illiquid: {failed+illiquid}")
        
    #     raise RuntimeError(f"Failed after {attempts} attempts")
    
    # def _fetch_klines(self, symbol: str, interval: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    #     """Fetch crypto klines from Binance - migrated from data_preparation.py"""
    #     url = BASE_URL
    #     params = {
    #         "symbol": symbol,
    #         "interval": interval,
    #         "startTime": int(start.timestamp() * 1000),
    #         "endTime": int(end.timestamp() * 1000),
    #         "limit": 1000
    #     }
    #     all_data = []
    #     while True:
    #         resp = requests.get(url, params=params)
    #         data = resp.json()
    #         if not isinstance(data, list):
    #             return None
    #         all_data.extend(data)
    #         if len(data) < 1000:
    #             break
    #         params["startTime"] = data[-1][6]
    #     if not all_data:
    #         return None
    #     df = pd.DataFrame(all_data, columns=[
    #         "openTime","open","high","low","close","volume",
    #         "closeTime","qav","trades","tbbav","tbqav","ignore"
    #     ])
    #     df["date"] = pd.to_datetime(df["openTime"], unit="ms")
    #     df = df[["date","open","high","low","close","volume"]]
    #     df = df.astype({"open":float,"high":float,"low":float,"close":float,"volume":float})
    #     return df
    def _create_crypto_dataset(self) -> pd.DataFrame:
        """
        Create crypto dataset using chunked download.
        REPLACES the old _create_crypto_dataset() method.
        """
        # Use chunked download instead of random sampling
        raw_data = self._download_crypto_chunked(
            symbols=CRYPTO_TICKERS,
            start_date=CRYPTO_START_DATE,
            end_date=CRYPTO_END_DATE,
            interval="15m",
            chunk_days=30
        )
        
        # Process same as before - these steps remain unchanged
        with_indicators = self._add_technical_indicators(raw_data)
        normalized = self._normalize_features(with_indicators)
        cleaned = self._clean_data(normalized)
        
        return cleaned


    def _download_crypto_chunked(
            self,
            symbols: List[str],
            start_date: datetime,
            end_date: datetime,
            interval: str = "15m",
            chunk_days: int = 30
        ) -> pd.DataFrame:
        """
        Download crypto data in chunks with validation and resume capability.
        Returns DataFrame with columns: ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        """
        logger.info(f"📥 Starting chunked download for {len(symbols)} tickers")
        logger.info(f"   Date range: {start_date.date()} to {end_date.date()}")
        logger.info(f"   Chunk size: {chunk_days} days")
        
        # Setup progress tracking
        progress_file = Path("environments/data/.crypto_download_progress.pkl")
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to resume from previous download
        all_chunks = []
        completed_chunks = set()
        
        if progress_file.exists():
            try:
                with open(progress_file, 'rb') as f:
                    saved_state = pickle.load(f)
                    all_chunks = saved_state.get('chunks', [])
                    completed_chunks = saved_state.get('completed', set())
                    logger.info(f"   📂 Resuming from checkpoint: {len(completed_chunks)} chunks completed")
            except Exception as e:
                logger.warning(f"   ⚠️  Could not load progress file: {e}")
        
        # Generate chunk boundaries
        chunks = []
        current_start = start_date
        
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=chunk_days), end_date)
            chunk_id = f"{current_start.strftime('%Y%m%d')}_{current_end.strftime('%Y%m%d')}"
            
            if chunk_id not in completed_chunks:
                chunks.append({
                    'id': chunk_id,
                    'start': current_start,
                    'end': current_end
                })
            
            current_start = current_end
        
        total_chunks = len(chunks) + len(completed_chunks)
        logger.info(f"   📊 Total chunks: {total_chunks} ({len(chunks)} remaining)")
        
        # Download each chunk
        for i, chunk in enumerate(chunks):
            chunk_num = len(completed_chunks) + i + 1
            logger.info(f"\n📦 Chunk {chunk_num}/{total_chunks}: {chunk['start'].date()} to {chunk['end'].date()}")
            
            chunk_data = self._download_chunk(
                symbols=symbols,
                start_date=chunk['start'],
                end_date=chunk['end'],
                interval=interval
            )
            
            if chunk_data is None:
                logger.error(f"   ❌ Failed to download chunk {chunk['id']}")
                logger.error(f"   💾 Progress saved. You can resume by running again.")
                self._save_progress(progress_file, all_chunks, completed_chunks)
                raise RuntimeError(f"Failed to download chunk {chunk['id']}")
            
            # Validate chunk
            if not self._validate_chunk(chunk_data, symbols, chunk['start'], chunk['end'], interval):
                logger.error(f"   ❌ Chunk validation failed for {chunk['id']}")
                self._save_progress(progress_file, all_chunks, completed_chunks)
                raise RuntimeError(f"Chunk validation failed for {chunk['id']}")
            
            logger.info(f"   ✅ Chunk validated: {len(chunk_data)} rows")
            
            all_chunks.append(chunk_data)
            completed_chunks.add(chunk['id'])
            
            # Save progress
            self._save_progress(progress_file, all_chunks, completed_chunks)
            
            # Rate limiting between chunks
            if i < len(chunks) - 1:
                sleep_time = 2.5
                logger.info(f"   ⏱️  Sleeping {sleep_time}s (rate limiting)...")
                time.sleep(sleep_time)
        
        # Combine all chunks
        logger.info(f"\n🔗 Combining {len(all_chunks)} chunks...")
        combined_data = pd.concat(all_chunks, ignore_index=True)
        combined_data = combined_data.sort_values(['date', 'ticker']).reset_index(drop=True)
        
        # Final validation
        logger.info(f"📋 Final dataset: {combined_data.shape}")
        logger.info(f"   Date range: {combined_data['date'].min()} to {combined_data['date'].max()}")
        logger.info(f"   Tickers: {combined_data['ticker'].nunique()}")
        logger.info(f"   Days: {combined_data['date'].nunique()}")
        
        # Cleanup progress file on success
        if progress_file.exists():
            progress_file.unlink()
            logger.info(f"   🧹 Cleaned up progress file")
        
        return combined_data


    def _download_chunk(
            self,
            symbols: List[str],
            start_date: datetime,
            end_date: datetime,
            interval: str
        ) -> Optional[pd.DataFrame]:
        """
        Download a single chunk of data for all symbols.
        Returns DataFrame or None if failed.
        """
        all_dfs = []
        failed = []
        
        for sym in symbols:
            df = self._fetch_klines_robust(sym, interval, start_date, end_date)
            
            if df is None or len(df) == 0:
                failed.append(sym)
                logger.warning(f"      ⚠️  {sym}: No data returned")
                continue
            
            # Check for sufficient volume
            if df["volume"].sum() <= 0:
                failed.append(sym)
                logger.warning(f"      ⚠️  {sym}: Zero volume")
                continue
            
            df["ticker"] = sym
            all_dfs.append(df)
            
            # Small delay between tickers
            time.sleep(0.1)
        
        if failed:
            logger.warning(f"      Failed/illiquid tickers: {', '.join(failed)}")
            return None
        
        if not all_dfs:
            return None
        
        # Combine all tickers
        chunk_df = pd.concat(all_dfs, ignore_index=True)
        return chunk_df


    def _fetch_klines_robust(
            self,
            symbol: str,
            interval: str,
            start: datetime,
            end: datetime,
            max_retries: int = 3
        ) -> Optional[pd.DataFrame]:
        """
        Fetch klines with retry logic and better error handling.
        Returns DataFrame with columns: ['date', 'open', 'high', 'low', 'close', 'volume']
        """
        for attempt in range(max_retries):
            try:
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": int(start.timestamp() * 1000),
                    "endTime": int(end.timestamp() * 1000),
                    "limit": 1000
                }
                
                all_data = []
                
                while True:
                    resp = requests.get(BASE_URL, params=params, timeout=30)
                    
                    # Check for rate limiting
                    if resp.status_code == 429:
                        retry_after = int(resp.headers.get('Retry-After', 60))
                        logger.warning(f"      Rate limited. Waiting {retry_after}s...")
                        time.sleep(retry_after)
                        continue
                    
                    resp.raise_for_status()
                    data = resp.json()
                    
                    if not isinstance(data, list):
                        logger.warning(f"      Unexpected response format for {symbol}")
                        return None
                    
                    if len(data) == 0:
                        break
                    
                    all_data.extend(data)
                    
                    # Check if we got all data
                    if len(data) < 1000:
                        break
                    
                    # Update start time for next batch
                    params["startTime"] = data[-1][6] + 1  # closeTime + 1
                    
                    # Small delay between batches
                    time.sleep(0.05)
                
                if not all_data:
                    return None
                
                # Convert to DataFrame
                df = pd.DataFrame(all_data, columns=[
                    "openTime", "open", "high", "low", "close", "volume",
                    "closeTime", "qav", "trades", "tbbav", "tbqav", "ignore"
                ])
                
                # Process timestamps
                df["date"] = pd.to_datetime(df["openTime"], unit="ms")
                
                # Select and convert columns
                df = df[["date", "open", "high", "low", "close", "volume"]]
                df = df.astype({
                    "open": float,
                    "high": float,
                    "low": float,
                    "close": float,
                    "volume": float
                })
                
                return df
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"      Retry {attempt+1}/{max_retries} for {symbol} in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"      Failed to fetch {symbol} after {max_retries} attempts: {e}")
                    return None
            
            except Exception as e:
                logger.error(f"      Unexpected error fetching {symbol}: {e}")
                return None
        
        return None


    def _validate_chunk(self, chunk_data: pd.DataFrame, expected_symbols: List[str], start_date: datetime, end_date: datetime, interval: str) -> bool:
        """
        Validate that chunk has complete coverage.
        """
        # Check all symbols present
        present_symbols = set(chunk_data['ticker'].unique())
        expected_set = set(expected_symbols)
        
        if present_symbols != expected_set:
            missing = expected_set - present_symbols
            logger.error(f"      Missing symbols: {missing}")
            return False
        
        # Check date range coverage
        min_date = chunk_data['date'].min()
        max_date = chunk_data['date'].max()
        
        # Calculate expected number of candles
        duration = (end_date - start_date).total_seconds()
        
        if interval == "15m":
            interval_seconds = 15 * 60
        elif interval == "1h":
            interval_seconds = 60 * 60
        elif interval == "1d":
            interval_seconds = 24 * 60 * 60
        else:
            logger.warning(f"      Unknown interval: {interval}")
            return True  # Skip validation for unknown intervals
        
        expected_candles = int(duration / interval_seconds)
        actual_candles = len(chunk_data) // len(expected_symbols)
        
        # Allow some tolerance (98% coverage)
        coverage_ratio = actual_candles / expected_candles
        
        if coverage_ratio < 0.98:
            logger.warning(f"      Coverage: {coverage_ratio:.2%} ({actual_candles}/{expected_candles} candles)")
            logger.warning(f"      Date range: {min_date} to {max_date}")
            return False
        
        # Check for duplicates
        duplicates = chunk_data.duplicated(subset=['date', 'ticker']).sum()
        if duplicates > 0:
            logger.error(f"      Found {duplicates} duplicate rows")
            return False
        
        return True


    def _save_progress(self, progress_file: Path, chunks: List[pd.DataFrame], completed_ids: set):
        """Save download progress for resume capability."""
        try:
            with open(progress_file, 'wb') as f:
                pickle.dump({
                    'chunks': chunks,
                    'completed': completed_ids,
                    'timestamp': datetime.utcnow()
                }, f)
        except Exception as e:
            logger.warning(f"Could not save progress: {e}")

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators - migrated from data_preparation.py"""
        logger.info("Adding technical indicators...")
        
        results = []
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
            
            # === Basic Returns ===
            ticker_data['returns'] = ticker_data['close'].pct_change()
            ticker_data['log_returns'] = np.log(ticker_data['close'] / ticker_data['close'].shift(1))
            
            # === Moving Averages ===
            ticker_data['sma_5'] = ticker_data['close'].rolling(5).mean()
            ticker_data['sma_20'] = ticker_data['close'].rolling(20).mean()
            ticker_data['sma_50'] = ticker_data['close'].rolling(50).mean()
            
            # === Volatility Measures ===
            ticker_data['volatility_5d'] = ticker_data['returns'].rolling(5).std()
            ticker_data['volatility_20d'] = ticker_data['returns'].rolling(20).std()
            
            # === RSI (Relative Strength Index) ===
            delta = ticker_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean() 
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            ticker_data['rsi'] = 100 - (100 / (1 + rs))
            
            # === Bollinger Bands ===
            ticker_data['bb_middle'] = ticker_data['close'].rolling(20).mean()
            bb_std = ticker_data['close'].rolling(20).std()
            ticker_data['bb_upper'] = ticker_data['bb_middle'] + (bb_std * 2)
            ticker_data['bb_lower'] = ticker_data['bb_middle'] - (bb_std * 2)
            ticker_data['bb_position'] = (ticker_data['close'] - ticker_data['bb_lower']) / (ticker_data['bb_upper'] - ticker_data['bb_lower'])
            
            # === MACD ===
            exp1 = ticker_data['close'].ewm(span=12).mean()
            exp2 = ticker_data['close'].ewm(span=26).mean()
            ticker_data['macd'] = exp1 - exp2
            ticker_data['macd_signal'] = ticker_data['macd'].ewm(span=9).mean()
            ticker_data['macd_histogram'] = ticker_data['macd'] - ticker_data['macd_signal']
            
            # === Volume Weighted Average Price (VWAP) ===
            ticker_data['vwap'] = (ticker_data['close'] * ticker_data['volume']).rolling(20).sum() / ticker_data['volume'].rolling(20).sum()
            
            # === Price Position ===
            ticker_data['high_low_pct'] = (ticker_data['close'] - ticker_data['low']) / (ticker_data['high'] - ticker_data['low'])
            
            results.append(ticker_data)
        
        combined = pd.concat(results, ignore_index=True)
        
        # === Market-Wide Features ===
        market_data = combined.groupby('date')['returns'].agg(['mean', 'std']).reset_index()
        market_data = market_data.rename(columns={'mean': 'market_return', 'std': 'market_volatility'})
        
        combined = combined.merge(market_data, on='date', how='left')
        combined['excess_returns'] = combined['returns'] - combined['market_return']
        
        # Market momentum
        market_data['market_momentum'] = market_data['market_return'].rolling(10).mean()
        combined = combined.merge(market_data[['date', 'market_momentum']], on='date', how='left')
        
        logger.info(f"Added technical indicators: {combined.shape}")
        return combined
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features for RL training - migrated from data_preparation.py"""
        logger.info("Normalizing features...")
        
        results = []
        
        # Define feature categories
        price_features = ['open', 'high', 'low', 'close', 'adj_close', 'sma_5', 'sma_20', 'sma_50',
                         'bb_upper', 'bb_lower', 'bb_middle', 'vwap']
        bounded_features = ['rsi', 'bb_position', 'high_low_pct']
        volatility_features = ['volatility_5d', 'volatility_20d', 'market_volatility']
        volume_features = ['volume']
        unbounded_features = ['macd', 'macd_signal', 'macd_histogram']
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()
            
            # === MinMax Normalization for Price Features ===
            for feature in price_features:
                if feature in ticker_data.columns:
                    min_val = ticker_data[feature].min()
                    max_val = ticker_data[feature].max()
                    if max_val > min_val:
                        ticker_data[f'{feature}_norm'] = (ticker_data[feature] - min_val) / (max_val - min_val)
                    else:
                        ticker_data[f'{feature}_norm'] = 0.5
            
            # === Scale Bounded Features to [-1, 1] ===
            for feature in bounded_features:
                if feature in ticker_data.columns:
                    if feature == 'rsi':
                        ticker_data[f'{feature}_norm'] = (ticker_data[feature] - 50) / 50
                    else:
                        ticker_data[f'{feature}_norm'] = (ticker_data[feature] - 0.5) * 2
            
            # === Robust Standardization for Volatility ===
            for feature in volatility_features:
                if feature in ticker_data.columns:
                    median_val = ticker_data[feature].median()
                    mad = (ticker_data[feature] - median_val).abs().median()
                    if mad > 0:
                        ticker_data[f'{feature}_norm'] = (ticker_data[feature] - median_val) / (1.4826 * mad)
                    else:
                        ticker_data[f'{feature}_norm'] = 0.0
            
            # === Log-transform + Standardization for Volume ===
            for feature in volume_features:
                if feature in ticker_data.columns:
                    log_volume = np.log1p(ticker_data[feature])
                    mean_val = log_volume.mean()
                    std_val = log_volume.std()
                    if std_val > 0:
                        ticker_data[f'{feature}_norm'] = (log_volume - mean_val) / std_val
                    else:
                        ticker_data[f'{feature}_norm'] = 0.0
            
            # === Z-score for Unbounded Features ===
            for feature in unbounded_features:
                if feature in ticker_data.columns:
                    mean_val = ticker_data[feature].mean()
                    std_val = ticker_data[feature].std()
                    if std_val > 0:
                        ticker_data[f'{feature}_norm'] = (ticker_data[feature] - mean_val) / std_val
                    else:
                        ticker_data[f'{feature}_norm'] = 0.0
            
            results.append(ticker_data)
        
        combined = pd.concat(results, ignore_index=True)
        
        # === Normalize Market-Wide Features ===
        market_features = ['market_return', 'excess_returns', 'market_momentum']
        for feature in market_features:
            if feature in combined.columns:
                mean_val = combined[feature].mean()
                std_val = combined[feature].std()
                if std_val > 0:
                    combined[f'{feature}_norm'] = (combined[feature] - mean_val) / std_val
                else:
                    combined[f'{feature}_norm'] = 0.0
        
        normalized_features = [col for col in combined.columns if col.endswith('_norm')]
        logger.info(f"Normalized {len(normalized_features)} features")
        
        return combined
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data - migrated from data_preparation.py"""
        logger.info("Cleaning data...")
        
        initial_rows = len(df)
        
        # Remove rows with missing essential data
        df = df.dropna(subset=['date', 'ticker', 'close'])
        
        # Handle rectangular structure
        date_counts = df.groupby('date').size()
        expected_tickers = df['ticker'].nunique()
        irregular_dates = date_counts[date_counts != expected_tickers]
        
        if len(irregular_dates) > 0:
            logger.warning(f"Found {len(irregular_dates)} dates with irregular ticker counts")
            valid_dates = date_counts[date_counts == expected_tickers].index
            df = df[df['date'].isin(valid_dates)]
        
        # Generate returns if missing
        if 'returns' not in df.columns or df['returns'].isnull().any():
            df = df.sort_values(['ticker', 'date'])
            df['returns'] = df.groupby('ticker')['close'].pct_change()
        
        # Final cleanup
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Handle remaining NaNs
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df = df.copy()
        
        for col in numeric_columns:
            if col.endswith('_norm'):
                df.loc[:, col] = df[col].fillna(0.0)
            elif 'rsi' in col.lower():
                df.loc[:, col] = df[col].fillna(50.0)
            elif 'volatility' in col.lower():
                df.loc[:, col] = df[col].fillna(0.01)
            elif col in ['bb_position', 'high_low_pct']:
                df.loc[:, col] = df[col].fillna(0.5)
            elif col == 'returns':
                df.loc[:, col] = df[col].fillna(0.0)
            else:
                df.loc[:, col] = df[col].fillna(df[col].median())
        
        # Remove any remaining rows with NaNs
        df = df.dropna()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date', 'ticker'])
        
        # Ensure proper data types
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
        
        logger.info(f"Cleaned data: {len(df)} rows (removed {initial_rows - len(df)})")
        
        return df


# ========== CONVENIENCE FUNCTIONS FOR BACKWARD COMPATIBILITY ==========

def create_split_datasets(data_path: str, train_end: str = '2015-12-31', 
                         val_end: str = '2020-12-31', proportional: bool = False,
                         proportions: Tuple[float, float, float] = (0.7, 0.2, 0.1)) -> Dict[str, DatasetSplit]:
    """Backward compatibility function - now uses PortfolioDataset internally"""
    # Determine asset class from path
    asset_class = "crypto" if "crypto" in data_path else "sp500"
    
    dataset = PortfolioDataset(
        asset_class=asset_class,
        data_path=data_path,
        train_end=train_end,
        val_end=val_end,
        proportional=proportional,
        proportions=proportions
    )
    
    return dataset.get_all_splits()


def create_dataset(output_path: str, tickers: List[str] = None, 
                  start_date: str = '1990-02-16', end_date: str = '2025-01-01',
                  force_recreate: bool = False) -> str:
    """Backward compatibility function for SP500 dataset creation"""
    asset_class = "sp500"
    
    dataset = PortfolioDataset(
        asset_class=asset_class,
        data_path=output_path,
        force_recreate=force_recreate
    )
    
    return output_path


def create_crypto_dataset(output_path: str, tickers: List[str] = None,
                         target_rows: int = 263520, days: int = 92,
                         interval: str = "15m", force_recreate: bool = False) -> str:
    """Backward compatibility function for crypto dataset creation"""
    asset_class = "crypto"
    
    dataset = PortfolioDataset(
        asset_class=asset_class,
        data_path=output_path,
        force_recreate=force_recreate
    )
    
    return output_path


def load_dataset(data_path: str) -> pd.DataFrame:
    """Backward compatibility function for loading datasets"""
    asset_class = "crypto" if "crypto" in data_path else "sp500"
    
    dataset = PortfolioDataset(asset_class=asset_class, data_path=data_path)
    return dataset.full_data


def get_dataset_info(data_path: str) -> Dict:
    """Get information about a dataset without loading it fully"""
    if not Path(data_path).exists():
        return {"exists": False, "error": "Dataset file not found"}
    
    try:
        asset_class = "crypto" if "crypto" in data_path else "sp500"
        dataset = PortfolioDataset(asset_class=asset_class, data_path=data_path)
        
        normalized_features = [col for col in dataset.full_data.columns if col.endswith('_norm')]
        technical_features = [col for col in dataset.full_data.columns if col not in 
                            ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]
        
        info = {
            "exists": True,
            "path": data_path,
            "file_size_mb": Path(data_path).stat().st_size / (1024 * 1024),
            "shape": dataset.full_data.shape,
            "date_range": {
                "start": str(dataset.full_data['date'].min().date()),
                "end": str(dataset.full_data['date'].max().date())
            },
            "tickers": {
                "count": dataset.full_data['ticker'].nunique(),
                "list": sorted(dataset.full_data['ticker'].unique().tolist())
            },
            "features": {
                "total": len(dataset.full_data.columns),
                "normalized": len(normalized_features),
                "technical": len(technical_features)
            },
            "columns": dataset.full_data.columns.tolist()
        }
        
        return info
        
    except Exception as e:
        return {"exists": True, "error": str(e)}