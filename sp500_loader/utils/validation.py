# validation.py
"""
Comprehensive data validation utilities for SP500 data loader.
Validates data quality, structure, and consistency for portfolio optimization.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional


def validate_panel_data(panel_df: pd.DataFrame, verbose: bool = True) -> Dict:
    """
    Comprehensive validation of panel data structure and quality.
    
    Parameters:
    -----------
    panel_df : pd.DataFrame
        Multi-index DataFrame from sp500_loader
    verbose : bool
        Whether to print detailed validation results
        
    Returns:
    --------
    dict: Comprehensive validation results and statistics
    """
    
    if verbose:
        print("=== SP500 DATA VALIDATION REPORT ===")
    
    results = {
        'validation_timestamp': datetime.now(),
        'overall_status': 'UNKNOWN',
        'errors': [],
        'warnings': [],
        'statistics': {},
        'recommendations': []
    }
    
    try:
        # 1. STRUCTURE VALIDATION
        if verbose:
            print("\n1. STRUCTURE VALIDATION")
        
        structure_results = _validate_structure(panel_df, verbose)
        results.update(structure_results)
        
        # 2. DATA QUALITY VALIDATION
        if verbose:
            print("\n2. DATA QUALITY VALIDATION")
        
        quality_results = _validate_data_quality(panel_df, verbose)
        results['statistics'].update(quality_results['statistics'])
        results['warnings'].extend(quality_results['warnings'])
        
        # 3. TEMPORAL VALIDATION
        if verbose:
            print("\n3. TEMPORAL VALIDATION")
        
        temporal_results = _validate_temporal_consistency(panel_df, verbose)
        results['statistics'].update(temporal_results['statistics'])
        results['warnings'].extend(temporal_results['warnings'])
        
        # 4. MARKET DATA VALIDATION
        if verbose:
            print("\n4. MARKET DATA VALIDATION")
        
        market_results = _validate_market_data(panel_df, verbose)
        results['statistics'].update(market_results['statistics'])
        results['warnings'].extend(market_results['warnings'])
        
        # 5. PORTFOLIO READINESS VALIDATION
        if verbose:
            print("\n5. PORTFOLIO OPTIMIZATION READINESS")
        
        portfolio_results = _validate_portfolio_readiness(panel_df, verbose)
        results['statistics'].update(portfolio_results['statistics'])
        results['recommendations'].extend(portfolio_results['recommendations'])
        
        # Determine overall status
        if len(results['errors']) == 0:
            if len(results['warnings']) == 0:
                results['overall_status'] = 'EXCELLENT'
            elif len(results['warnings']) <= 3:
                results['overall_status'] = 'GOOD'
            else:
                results['overall_status'] = 'ACCEPTABLE'
        else:
            results['overall_status'] = 'NEEDS_ATTENTION'
        
        if verbose:
            _print_summary(results)
            
    except Exception as e:
        results['errors'].append(f"Validation failed: {str(e)}")
        results['overall_status'] = 'ERROR'
        if verbose:
            print(f"❌ Validation error: {e}")
    
    return results


def _validate_structure(panel_df: pd.DataFrame, verbose: bool) -> Dict:
    """Validate the basic structure of the panel data."""
    
    results = {'errors': [], 'statistics': {}}
    
    # Check if DataFrame exists and has data
    if panel_df is None or len(panel_df) == 0:
        results['errors'].append("DataFrame is empty or None")
        return results
    
    # Check MultiIndex structure
    if not isinstance(panel_df.index, pd.MultiIndex):
        results['errors'].append("Expected MultiIndex, got single index")
    else:
        if verbose:
            print("✓ MultiIndex structure correct")
    
    # Check index names
    expected_index_names = ['date', 'ticker']
    if panel_df.index.names != expected_index_names:
        results['errors'].append(f"Expected index names {expected_index_names}, got {panel_df.index.names}")
    else:
        if verbose:
            print("✓ Index names correct")
    
    # Check required columns
    required_columns = ['adj_close', 'volume', 'is_active']
    missing_columns = [col for col in required_columns if col not in panel_df.columns]
    if missing_columns:
        results['errors'].append(f"Missing required columns: {missing_columns}")
    else:
        if verbose:
            print("✓ All required columns present")
    
    # Check data types
    if 'adj_close' in panel_df.columns:
        if not pd.api.types.is_numeric_dtype(panel_df['adj_close']):
            results['errors'].append("adj_close should be numeric")
        else:
            if verbose:
                print("✓ adj_close data type correct")
    
    if 'is_active' in panel_df.columns:
        if panel_df['is_active'].dtype not in ['int64', 'int32', 'bool', 'int8']:
            results['errors'].append(f"is_active should be int/bool, got {panel_df['is_active'].dtype}")
        else:
            if verbose:
                print("✓ is_active data type correct")
    
    # Basic statistics
    results['statistics'] = {
        'total_rows': len(panel_df),
        'num_dates': len(panel_df.index.get_level_values('date').unique()) if isinstance(panel_df.index, pd.MultiIndex) else 0,
        'num_tickers': len(panel_df.index.get_level_values('ticker').unique()) if isinstance(panel_df.index, pd.MultiIndex) else 0,
        'date_range': (
            panel_df.index.get_level_values('date').min(),
            panel_df.index.get_level_values('date').max()
        ) if isinstance(panel_df.index, pd.MultiIndex) else (None, None)
    }
    
    if verbose:
        stats = results['statistics']
        print(f"📊 Basic Statistics:")
        print(f"   Total rows: {stats['total_rows']:,}")
        print(f"   Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        print(f"   Unique dates: {stats['num_dates']:,}")
        print(f"   Unique tickers: {stats['num_tickers']}")
    
    return results


def _validate_data_quality(panel_df: pd.DataFrame, verbose: bool) -> Dict:
    """Validate data quality metrics."""
    
    results = {'warnings': [], 'statistics': {}}
    
    # Price data quality
    price_col = 'adj_close'
    if price_col in panel_df.columns:
        prices = panel_df[price_col]
        
        # Missing data analysis
        total_possible = len(panel_df)
        missing_count = prices.isna().sum()
        missing_rate = missing_count / total_possible
        
        # Active vs price availability
        if 'is_active' in panel_df.columns:
            active_points = (panel_df['is_active'] == 1).sum()
            prices_when_active = prices[panel_df['is_active'] == 1]
            missing_when_active = prices_when_active.isna().sum()
            active_coverage = (active_points - missing_when_active) / active_points if active_points > 0 else 0
        else:
            active_coverage = None
            active_points = 0
        
        # Price reasonableness checks
        valid_prices = prices.dropna()
        if len(valid_prices) > 0:
            negative_prices = (valid_prices < 0).sum()
            zero_prices = (valid_prices == 0).sum()
            extreme_low = (valid_prices < 0.01).sum()
            extreme_high = (valid_prices > 100000).sum()
            
            if negative_prices > 0:
                results['warnings'].append(f"Found {negative_prices} negative prices")
            if zero_prices > 0:
                results['warnings'].append(f"Found {zero_prices} zero prices")
            if extreme_low > negative_prices:
                results['warnings'].append(f"Found {extreme_low - negative_prices} suspiciously low prices (<$0.01)")
            if extreme_high > 0:
                results['warnings'].append(f"Found {extreme_high} extremely high prices (>$100k)")
        
        results['statistics'].update({
            'total_data_points': total_possible,
            'missing_data_points': missing_count,
            'missing_rate': missing_rate,
            'active_data_points': active_points,
            'active_coverage': active_coverage,
            'price_range': (valid_prices.min(), valid_prices.max()) if len(valid_prices) > 0 else (None, None),
            'median_price': valid_prices.median() if len(valid_prices) > 0 else None
        })
        
        if verbose:
            print(f"📊 Data Quality Metrics:")
            print(f"   Missing data rate: {missing_rate:.1%}")
            if active_coverage is not None:
                print(f"   Coverage when active: {active_coverage:.1%}")
            if len(valid_prices) > 0:
                print(f"   Price range: ${valid_prices.min():.2f} - ${valid_prices.max():.2f}")
                print(f"   Median price: ${valid_prices.median():.2f}")
        
        # Quality warnings
        if missing_rate > 0.3:
            results['warnings'].append(f"High missing data rate: {missing_rate:.1%}")
        if active_coverage is not None and active_coverage < 0.95:
            results['warnings'].append(f"Low coverage when stocks are active: {active_coverage:.1%}")
    
    return results


def _validate_temporal_consistency(panel_df: pd.DataFrame, verbose: bool) -> Dict:
    """Validate temporal aspects of the data."""
    
    results = {'warnings': [], 'statistics': {}}
    
    if not isinstance(panel_df.index, pd.MultiIndex):
        return results
    
    dates = panel_df.index.get_level_values('date')
    
    # Check for date sorting
    if not dates.is_monotonic_increasing:
        results['warnings'].append("Dates are not sorted in ascending order")
    
    # Analyze date frequency
    unique_dates = dates.unique()
    if len(unique_dates) > 1:
        date_diffs = pd.Series(unique_dates).diff().dropna()
        
        # Business day analysis
        business_days = pd.bdate_range(unique_dates.min(), unique_dates.max())
        business_day_coverage = len(unique_dates) / len(business_days)
        
        # Gap analysis
        expected_diff = pd.Timedelta(days=1)
        large_gaps = date_diffs[date_diffs > pd.Timedelta(days=7)]  # Gaps > 1 week
        
        results['statistics'].update({
            'date_span_days': (unique_dates.max() - unique_dates.min()).days,
            'unique_dates_count': len(unique_dates),
            'business_day_coverage': business_day_coverage,
            'large_gaps_count': len(large_gaps),
            'max_gap_days': large_gaps.max().days if len(large_gaps) > 0 else 0
        })
        
        if verbose:
            print(f"📊 Temporal Analysis:")
            print(f"   Date span: {results['statistics']['date_span_days']} days")
            print(f"   Business day coverage: {business_day_coverage:.1%}")
            print(f"   Large gaps (>7 days): {len(large_gaps)}")
        
        # Temporal warnings
        if business_day_coverage < 0.8:
            results['warnings'].append(f"Low business day coverage: {business_day_coverage:.1%}")
        if len(large_gaps) > 10:
            results['warnings'].append(f"Many large date gaps: {len(large_gaps)}")
    
    return results


def _validate_market_data(panel_df: pd.DataFrame, verbose: bool) -> Dict:
    """Validate market-specific data patterns."""
    
    results = {'warnings': [], 'statistics': {}}
    
    if 'adj_close' not in panel_df.columns or 'is_active' not in panel_df.columns:
        return results
    
    # Analyze per-ticker statistics
    ticker_stats = []
    
    for ticker in panel_df.index.get_level_values('ticker').unique():
        ticker_data = panel_df.xs(ticker, level='ticker')
        
        active_periods = ticker_data['is_active'].sum()
        total_periods = len(ticker_data)
        price_availability = ticker_data['adj_close'].notna().sum()
        
        if active_periods > 0:
            prices = ticker_data['adj_close'].dropna()
            if len(prices) > 1:
                returns = prices.pct_change().dropna()
                
                ticker_stats.append({
                    'ticker': ticker,
                    'active_periods': active_periods,
                    'total_periods': total_periods,
                    'activity_rate': active_periods / total_periods,
                    'price_availability': price_availability,
                    'avg_price': prices.mean(),
                    'price_volatility': returns.std() if len(returns) > 0 else np.nan,
                    'max_return': returns.max() if len(returns) > 0 else np.nan,
                    'min_return': returns.min() if len(returns) > 0 else np.nan
                })
    
    if ticker_stats:
        ticker_df = pd.DataFrame(ticker_stats)
        
        # Market-level statistics
        results['statistics'].update({
            'tickers_analyzed': len(ticker_stats),
            'avg_activity_rate': ticker_df['activity_rate'].mean(),
            'min_activity_rate': ticker_df['activity_rate'].min(),
            'avg_volatility': ticker_df['price_volatility'].mean(),
            'max_daily_return': ticker_df['max_return'].max(),
            'min_daily_return': ticker_df['min_return'].min()
        })
        
        if verbose:
            print(f"📊 Market Data Analysis:")
            print(f"   Tickers analyzed: {len(ticker_stats)}")
            print(f"   Average activity rate: {ticker_df['activity_rate'].mean():.1%}")
            print(f"   Average volatility: {ticker_df['price_volatility'].mean():.1%}")
        
        # Market warnings
        low_activity_tickers = ticker_df[ticker_df['activity_rate'] < 0.5]
        if len(low_activity_tickers) > 0:
            results['warnings'].append(f"{len(low_activity_tickers)} tickers have <50% activity rate")
        
        extreme_volatility = ticker_df[ticker_df['price_volatility'] > 0.1]  # >10% daily volatility
        if len(extreme_volatility) > 0:
            results['warnings'].append(f"{len(extreme_volatility)} tickers have extreme volatility (>10%)")
    
    return results


def _validate_portfolio_readiness(panel_df: pd.DataFrame, verbose: bool) -> Dict:
    """Validate readiness for portfolio optimization tasks."""
    
    results = {'recommendations': [], 'statistics': {}}
    
    # Simulate typical training split
    if isinstance(panel_df.index, pd.MultiIndex):
        dates = panel_df.index.get_level_values('date').unique()
        
        if len(dates) > 500:  # Only meaningful for reasonable-sized datasets
            # Simulate 60/20/20 split
            train_end_idx = int(len(dates) * 0.6)
            val_end_idx = int(len(dates) * 0.8)
            
            train_dates = dates[:train_end_idx]
            val_dates = dates[train_end_idx:val_end_idx]
            test_dates = dates[val_end_idx:]
            
            # Analyze availability in each split
            for split_name, split_dates in [('train', train_dates), ('val', val_dates), ('test', test_dates)]:
                split_data = panel_df[panel_df.index.get_level_values('date').isin(split_dates)]
                
                if len(split_data) > 0:
                    active_data = split_data[split_data['is_active'] == 1]
                    
                    # Count stocks with sufficient data in this split
                    min_days = 30  # Minimum for episode creation
                    ticker_counts = active_data.groupby('ticker').size()
                    sufficient_tickers = (ticker_counts >= min_days).sum()
                    
                    results['statistics'][f'{split_name}_sufficient_tickers'] = sufficient_tickers
                    results['statistics'][f'{split_name}_total_days'] = len(split_dates)
            
            if verbose:
                print(f"📊 Portfolio Optimization Readiness:")
                for split in ['train', 'val', 'test']:
                    key = f'{split}_sufficient_tickers'
                    if key in results['statistics']:
                        print(f"   {split.capitalize()}: {results['statistics'][key]} tickers with ≥30 days")
            
            # Recommendations
            train_tickers = results['statistics'].get('train_sufficient_tickers', 0)
            if train_tickers < 20:
                results['recommendations'].append("Consider longer training period - only {train_tickers} tickers have sufficient data")
            elif train_tickers < 50:
                results['recommendations'].append("Training data acceptable but consider more history for robust portfolio optimization")
            
            # Episode feasibility check
            if train_tickers >= 10:
                results['recommendations'].append("✓ Sufficient tickers for episode-based training")
            else:
                results['recommendations'].append("⚠️ May have difficulty creating episodes - consider relaxing filtering criteria")
    
    return results


def _print_summary(results: Dict) -> None:
    """Print a summary of validation results."""
    
    status_emojis = {
        'EXCELLENT': '🟢',
        'GOOD': '🟡', 
        'ACCEPTABLE': '🟠',
        'NEEDS_ATTENTION': '🔴',
        'ERROR': '❌'
    }
    
    print(f"\n{'='*50}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*50}")
    print(f"Overall Status: {status_emojis.get(results['overall_status'], '❓')} {results['overall_status']}")
    
    if results['errors']:
        print(f"\n❌ ERRORS ({len(results['errors'])}):")
        for error in results['errors']:
            print(f"   • {error}")
    
    if results['warnings']:
        print(f"\n⚠️  WARNINGS ({len(results['warnings'])}):")
        for warning in results['warnings']:
            print(f"   • {warning}")
    
    if results['recommendations']:
        print(f"\n💡 RECOMMENDATIONS ({len(results['recommendations'])}):")
        for rec in results['recommendations']:
            print(f"   • {rec}")
    
    print(f"\nValidation completed at: {results['validation_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")


def quick_validate(panel_df: pd.DataFrame) -> bool:
    """
    Quick validation that returns True/False for basic pipeline checks.
    
    Parameters:
    -----------
    panel_df : pd.DataFrame
        Panel data to validate
        
    Returns:
    --------
    bool: True if data passes basic checks for pipeline use
    """
    
    try:
        results = validate_panel_data(panel_df, verbose=False)
        return len(results['errors']) == 0
    except:
        return False


def validate_splits(loader, verbose: bool = True) -> Dict:
    """
    Validate split loader configuration and episode generation.
    
    Parameters:
    -----------
    loader : QuickSplitLoader
        The split loader to validate
    verbose : bool
        Whether to print validation details
        
    Returns:
    --------
    dict: Validation results for splits and episodes
    """
    
    if verbose:
        print("=== SPLIT VALIDATION ===")
    
    results = {
        'split_validation': {},
        'episode_validation': {},
        'recommendations': []
    }
    
    try:
        splits = loader.splits
        
        # Validate temporal splits
        for split_name in ['train', 'val', 'test']:
            split_data = splits['temporal_splits'][split_name]
            prices = split_data['prices']
            
            results['split_validation'][split_name] = {
                'date_range': (prices.index.min(), prices.index.max()),
                'days': len(prices),
                'tickers': len(prices.columns),
                'data_coverage': prices.notna().mean().mean()
            }
        
        # Validate episodes
        for split_name in ['train', 'val', 'test']:
            episodes = splits['episodic_data'][split_name]
            
            if episodes:
                episode_lengths = [len(ep['prices']) for ep in episodes]
                difficulty_dist = pd.Series([ep.get('episode_difficulty', 'normal') for ep in episodes]).value_counts()
                
                results['episode_validation'][split_name] = {
                    'episode_count': len(episodes),
                    'avg_length': np.mean(episode_lengths),
                    'difficulty_distribution': difficulty_dist.to_dict()
                }
            else:
                results['episode_validation'][split_name] = {
                    'episode_count': 0,
                    'issue': 'No episodes generated'
                }
        
        if verbose:
            for split_name in ['train', 'val', 'test']:
                split_info = results['split_validation'][split_name]
                episode_info = results['episode_validation'][split_name]
                
                print(f"\n{split_name.upper()}:")
                print(f"   Date range: {split_info['date_range'][0].date()} to {split_info['date_range'][1].date()}")
                print(f"   Days: {split_info['days']}, Tickers: {split_info['tickers']}")
                print(f"   Episodes: {episode_info['episode_count']}")
        
        # Generate recommendations
        train_episodes = results['episode_validation']['train']['episode_count']
        if train_episodes == 0:
            results['recommendations'].append("No training episodes - consider relaxing episode_length or min_history_days")
        elif train_episodes < 50:
            results['recommendations'].append("Few training episodes - consider longer training period or shorter episodes")
        
    except Exception as e:
        results['error'] = str(e)
    
    return results


# Example usage and testing
if __name__ == "__main__":
    print("SP500 Data Validation Utilities")
    print("Import this module to use validation functions:")
    print("  from sp500_loader.utils.validation import validate_panel_data, quick_validate")
    print("  results = validate_panel_data(your_panel_df)")
    print("  is_valid = quick_validate(your_panel_df)")