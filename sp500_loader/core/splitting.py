# splitting.py
"""
Runtime data splitting and episode creation for S&P 500 portfolio optimization.
Designed for quick deployment on remote servers without disk storage.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import random


def prepare_data_for_splitting(panel_df):
    """
    Convert multi-index panel data to the format expected by splitting functions.
    
    Parameters:
    -----------
    panel_df : pd.DataFrame
        Multi-index DataFrame from sp500_loader with (date, ticker) index
        
    Returns:
    --------
    tuple: (price_data, is_active_data) as expected by create_temporal_splits
    """
    
    # Extract price data and pivot to get dates as index, tickers as columns
    price_data = panel_df['adj_close'].unstack(level='ticker')
    
    # Extract is_active data and pivot similarly
    is_active_data = panel_df['is_active'].unstack(level='ticker').astype(bool)
    
    print(f"Prepared data shape: {price_data.shape}")
    print(f"Date range: {price_data.index[0]} to {price_data.index[-1]}")
    print(f"Number of tickers: {price_data.columns.nunique()}")
    
    return price_data, is_active_data


def create_temporal_splits(price_data, is_active_data, 
                          train_end='2015-12-31', 
                          val_end='2018-12-31',
                          min_history_days=252):
    """
    Create temporal train/validation/test splits for portfolio optimization.
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        Adjusted close prices with dates as index and tickers as columns
    is_active_data : pd.DataFrame
        Boolean mask indicating if stock data is available
    train_end : str
        End date for training period
    val_end : str
        End date for validation period
    min_history_days : int
        Minimum number of days a stock must have data to be included
    
    Returns:
    --------
    dict with train/val/test DataFrames and metadata
    """
    
    # Convert dates
    train_end_date = pd.to_datetime(train_end)
    val_end_date = pd.to_datetime(val_end)
    
    # Create splits
    train_mask = price_data.index <= train_end_date
    val_mask = (price_data.index > train_end_date) & (price_data.index <= val_end_date)
    test_mask = price_data.index > val_end_date
    
    # Get data for each split
    train_prices = price_data.loc[train_mask]
    val_prices = price_data.loc[val_mask]
    test_prices = price_data.loc[test_mask]
    
    train_active = is_active_data.loc[train_mask]
    val_active = is_active_data.loc[val_mask]
    test_active = is_active_data.loc[test_mask]
    
    # Filter stocks that have sufficient history in training
    # Note: We need to count only active days, not just non-null days
    valid_stocks = (train_active & train_prices.notna()).sum() >= min_history_days
    selected_tickers = valid_stocks[valid_stocks].index.tolist()
    
    print(f"Selected {len(selected_tickers)} stocks with at least {min_history_days} days of training data")
    print(f"Train period: {train_prices.index[0]} to {train_prices.index[-1]} ({len(train_prices)} days)")
    print(f"Val period: {val_prices.index[0]} to {val_prices.index[-1]} ({len(val_prices)} days)")
    print(f"Test period: {test_prices.index[0]} to {test_prices.index[-1]} ({len(test_prices)} days)")
    
    return {
        'train': {
            'prices': train_prices[selected_tickers],
            'is_active': train_active[selected_tickers]
        },
        'val': {
            'prices': val_prices[selected_tickers],
            'is_active': val_active[selected_tickers]
        },
        'test': {
            'prices': test_prices[selected_tickers],
            'is_active': test_active[selected_tickers]
        },
        'selected_tickers': selected_tickers,
        'dates': {
            'train_start': train_prices.index[0],
            'train_end': train_prices.index[-1],
            'val_start': val_prices.index[0],
            'val_end': val_prices.index[-1],
            'test_start': test_prices.index[0],
            'test_end': test_prices.index[-1]
        }
    }


def create_episode_based_splits(data_splits, episode_length=30, overlap=0):
    """
    Create episode-based datasets for meta-learning.
    Each episode represents a potential "task" or market regime.
    
    Parameters:
    -----------
    data_splits : dict
        Output from create_temporal_splits
    episode_length : int
        Number of trading days per episode
    overlap : int
        Number of overlapping days between episodes
    
    Returns:
    --------
    dict with episodic data for each split
    """
    
    def create_episodes(prices, is_active, episode_length, overlap):
        episodes = []
        stride = episode_length - overlap
        
        for start_idx in range(0, len(prices) - episode_length + 1, stride):
            end_idx = start_idx + episode_length
            
            episode_prices = prices.iloc[start_idx:end_idx]
            episode_active = is_active.iloc[start_idx:end_idx]
            
            # Calculate availability metrics for training insights
            valid_data_ratio = (episode_active & episode_prices.notna()).sum() / len(episode_prices)
            stocks_with_sufficient_data = (valid_data_ratio >= 0.6).sum()  # Lowered threshold
            total_stocks = len(episode_prices.columns)
            
            # More permissive thresholds to include market reality scenarios
            min_available_stocks = max(10, total_stocks * 0.3)  # At least 10 stocks or 30% of universe
            
            # Include episode if we have minimum viable stock universe
            if stocks_with_sufficient_data >= min_available_stocks:
                
                # Calculate additional metrics for the agent to learn from
                availability_stats = {
                    'avg_availability': valid_data_ratio.mean(),
                    'min_availability': valid_data_ratio.min(),
                    'stocks_available_ratio': stocks_with_sufficient_data / total_stocks,
                    'early_period_ratio': episode_active.iloc[:episode_length//3].sum().sum() / (total_stocks * episode_length//3),
                    'late_period_ratio': episode_active.iloc[-episode_length//3:].sum().sum() / (total_stocks * episode_length//3)
                }
                
                episodes.append({
                    'prices': episode_prices,
                    'is_active': episode_active,
                    'start_date': episode_prices.index[0],
                    'end_date': episode_prices.index[-1],
                    'returns': episode_prices.pct_change().iloc[1:],  # Daily returns
                    'valid_stocks': stocks_with_sufficient_data,
                    'availability_stats': availability_stats,
                    'episode_difficulty': 'high' if availability_stats['stocks_available_ratio'] < 0.7 else 'normal'
                })
        
        return episodes
    
    episodic_data = {}
    
    for split in ['train', 'val', 'test']:
        episodes = create_episodes(
            data_splits[split]['prices'],
            data_splits[split]['is_active'],
            episode_length,
            overlap
        )
        episodic_data[split] = episodes
        print(f"{split.capitalize()}: {len(episodes)} episodes of length {episode_length}")
    
    return episodic_data


class QuickSplitLoader:
    """
    Lightweight split loader for remote server deployment.
    Creates splits once and caches in memory.
    """
    
    def __init__(self, panel_df, train_end='2015-12-31', val_end='2018-12-31', 
                 min_history_days=252, episode_length=30, overlap=0):
        """
        Initialize and create splits immediately.
        
        Parameters:
        -----------
        panel_df : pd.DataFrame
            Multi-index DataFrame from sp500_loader
        train_end : str
            End date for training period
        val_end : str
            End date for validation period
        min_history_days : int
            Minimum trading days required for stock inclusion
        episode_length : int
            Number of trading days per episode
        overlap : int
            Number of overlapping days between episodes
        """
        
        print("=== INITIALIZING SPLIT LOADER ===")
        
        # Store config for reference
        self.config = {
            'train_end': train_end,
            'val_end': val_end,
            'min_history_days': min_history_days,
            'episode_length': episode_length,
            'overlap': overlap
        }
        
        # Create splits immediately
        print("Preparing data...")
        price_data, is_active_data = prepare_data_for_splitting(panel_df)
        
        print("Creating temporal splits...")
        temporal_splits = create_temporal_splits(
            price_data, is_active_data,
            train_end=train_end,
            val_end=val_end,
            min_history_days=min_history_days
        )
        
        print("Creating episodic splits...")
        episodic_data = create_episode_based_splits(
            temporal_splits, 
            episode_length=episode_length, 
            overlap=overlap
        )
        
        # Store in memory
        self._splits = {
            'temporal_splits': temporal_splits,
            'episodic_data': episodic_data,
            'metadata': {
                'total_tickers': len(price_data.columns),
                'selected_tickers': len(temporal_splits['selected_tickers']),
                'date_range': (price_data.index[0], price_data.index[-1]),
                'config': self.config
            }
        }
        
        print(f"✓ Split loader ready: {self._splits['metadata']['selected_tickers']} tickers, "
              f"{len(episodic_data['train'])} train episodes")
    
    @property
    def splits(self):
        """Access to all split data."""
        return self._splits
    
    def get_episodes(self, split_name):
        """Get episodes for a specific split."""
        return self._splits['episodic_data'][split_name]
    
    def get_episode_batch(self, split_name, batch_size=32, shuffle=True, difficulty_balance=False):
        """
        Get batches of episodes for training.
        
        Parameters:
        -----------
        split_name : str
            'train', 'val', or 'test'
        batch_size : int
            Number of episodes per batch
        shuffle : bool
            Whether to shuffle episodes
        difficulty_balance : bool
            Whether to balance episode difficulty (for training)
        
        Yields:
        -------
        list: Batch of episodes
        """
        episodes = self.get_episodes(split_name).copy()
        
        if shuffle:
            random.shuffle(episodes)
        
        # Optional difficulty balancing for training
        if split_name == 'train' and difficulty_balance:
            high_difficulty = [ep for ep in episodes if ep.get('episode_difficulty') == 'high']
            normal_difficulty = [ep for ep in episodes if ep.get('episode_difficulty') == 'normal']
            
            print(f"Episode difficulty distribution: {len(high_difficulty)} high, {len(normal_difficulty)} normal")
            
            high_per_batch = max(1, batch_size // 4)  # 25% high difficulty
            normal_per_batch = batch_size - high_per_batch
            
            high_idx, normal_idx = 0, 0
            while high_idx < len(high_difficulty) or normal_idx < len(normal_difficulty):
                batch = []
                
                # Add high difficulty episodes
                for _ in range(min(high_per_batch, len(high_difficulty) - high_idx)):
                    if high_idx < len(high_difficulty):
                        batch.append(high_difficulty[high_idx])
                        high_idx += 1
                
                # Add normal difficulty episodes  
                for _ in range(min(normal_per_batch, len(normal_difficulty) - normal_idx)):
                    if normal_idx < len(normal_difficulty):
                        batch.append(normal_difficulty[normal_idx])
                        normal_idx += 1
                
                if batch:
                    if shuffle:
                        random.shuffle(batch)
                    yield batch
        else:
            # Standard batching
            for i in range(0, len(episodes), batch_size):
                yield episodes[i:i + batch_size]
    
    def get_curriculum_batches(self, split_name='train', batch_size=32, start_difficulty='normal'):
        """
        Get curriculum learning batches, starting with easier episodes.
        
        Parameters:
        -----------
        split_name : str
            Split to use
        batch_size : int
            Batch size
        start_difficulty : str
            'normal' to start easy, 'mixed' for balanced, 'high' for hard start
        
        Yields:
        -------
        list: Batch of episodes with curriculum progression
        """
        episodes = self.get_episodes(split_name)
        
        if start_difficulty == 'normal':
            # Start with normal difficulty, gradually introduce high difficulty
            normal_episodes = [ep for ep in episodes if ep.get('episode_difficulty') == 'normal']
            high_episodes = [ep for ep in episodes if ep.get('episode_difficulty') == 'high']
            
            # First half: mostly normal episodes
            curriculum = normal_episodes + high_episodes[:len(high_episodes)//3]
            # Second half: mixed episodes
            curriculum.extend(high_episodes[len(high_episodes)//3:])
            
        elif start_difficulty == 'mixed':
            curriculum = episodes
        else:  # start_difficulty == 'high'
            # Start with challenging episodes (for advanced training)
            high_episodes = [ep for ep in episodes if ep.get('episode_difficulty') == 'high']
            normal_episodes = [ep for ep in episodes if ep.get('episode_difficulty') == 'normal']
            curriculum = high_episodes + normal_episodes
        
        for i in range(0, len(curriculum), batch_size):
            yield curriculum[i:i + batch_size]
    
    def print_summary(self):
        """Print summary of the loaded splits."""
        meta = self._splits['metadata']
        episodes = self._splits['episodic_data']
        
        print(f"\n=== SPLIT SUMMARY ===")
        print(f"Configuration: {self.config}")
        print(f"Selected {meta['selected_tickers']} out of {meta['total_tickers']} tickers")
        print(f"Date range: {meta['date_range'][0]} to {meta['date_range'][1]}")
        print(f"Episodes - Train: {len(episodes['train'])}, Val: {len(episodes['val'])}, Test: {len(episodes['test'])}")
        
        # Difficulty distribution
        for split_name in ['train', 'val', 'test']:
            eps = episodes[split_name]
            high_count = sum(1 for ep in eps if ep.get('episode_difficulty') == 'high')
            print(f"{split_name.capitalize()} difficulty: {high_count} high, {len(eps)-high_count} normal")


# Simplified usage for remote servers
def create_quick_loader(panel_df, **config):
    """
    One-line function to create a split loader.
    
    Parameters:
    -----------
    panel_df : pd.DataFrame
        Panel data from sp500_loader
    **config : dict
        Split configuration parameters
    
    Returns:
    --------
    QuickSplitLoader: Ready-to-use split loader
    """
    return QuickSplitLoader(panel_df, **config)


# Example usage for remote deployment
if __name__ == "__main__":
    # Quick setup for remote server
    from .loader import load_dataset
    
    print("=== QUICK REMOTE SERVER SETUP ===")
    
    # Load data
    panel_df = load_dataset('data/sp500_dataset.parquet')
    
    # Create splits in one line
    loader = create_quick_loader(
        panel_df,
        train_end='2015-12-31',
        val_end='2018-12-31',
        episode_length=30
    )
    
    # Print summary
    loader.print_summary()
    
    # Ready for training!
    print("\n=== READY FOR TRAINING ===")
    print("Example usage:")
    print("for epoch in range(num_epochs):")
    print("    for batch in loader.get_episode_batch('train', batch_size=32):")
    print("        # train on batch...")