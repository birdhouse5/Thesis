"""Test temporal split with comprehensive logging."""

import sys
import yaml
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.data.temporal_split import TemporalSplitter, TemporalSplitConfig
from src.utils.experiment_logger import create_experiment_logger


def load_config(config_path: str = 'configs/temporal_split_config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def calculate_data_metrics(data: pd.DataFrame) -> dict:
    """Calculate comprehensive data metrics."""
    metrics = {}
    
    # Basic shape
    metrics['n_days'] = len(data)
    metrics['n_features'] = data.shape[1]
    metrics['date_range'] = f"{data.index[0].date()} to {data.index[-1].date()}"
    
    # Missing data
    metrics['missing_values'] = data.isna().sum().sum()
    metrics['missing_percentage'] = (metrics['missing_values'] / data.size) * 100
    
    # Asset-specific metrics
    if 'Close' in data.columns.get_level_values(0):
        close_data = data['Close']
        
        # Returns statistics
        returns = close_data.pct_change()
        metrics['mean_daily_return'] = returns.mean().mean()
        metrics['mean_daily_volatility'] = returns.std().mean()
        metrics['max_daily_return'] = returns.max().max()
        metrics['min_daily_return'] = returns.min().min()
        
        # Correlation
        if close_data.shape[1] > 1:
            corr_matrix = close_data.pct_change().corr()
            metrics['mean_correlation'] = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean()
            metrics['max_correlation'] = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].max()
    
    return metrics


def visualize_splits(data: pd.DataFrame, splits: dict, episodes: dict, logger):
    """Visualize the temporal splits and episodes."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Data coverage
    colors = {'train': 'blue', 'val': 'orange', 'test': 'green'}
    
    for split_name, split_data in splits.items():
        if len(split_data) > 0:
            ax1.axvspan(split_data.index[0], split_data.index[-1], 
                       alpha=0.3, color=colors[split_name], label=split_name)
    
    # Add price data for context
    if 'Close' in data.columns:
        # Handle multi-asset case
        if isinstance(data['Close'], pd.DataFrame):
            for i, asset in enumerate(data['Close'].columns):
                price = data['Close'][asset]
                ax1.plot(data.index, price / price.iloc[0] * 100, 
                        alpha=0.7, label=f'{asset} (normalized)')
        else:
            price = data['Close']
            ax1.plot(data.index, price / price.iloc[0] * 100, 
                    'k-', alpha=0.7, label='Price (normalized)')
    
    ax1.set_ylabel('Normalized Price (%)')
    ax1.legend(loc='upper left')
    ax1.set_title('Temporal Data Splits with Price Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Episodes
    episode_y = 0
    task_colors = {}
    
    for split_name in ['train', 'val', 'test']:
        if split_name in episodes:
            for ep in episodes[split_name]:
                # Assign color to task
                if ep['task_id'] not in task_colors:
                    task_colors[ep['task_id']] = plt.cm.tab20(len(task_colors) % 20)
                
                ax2.barh(episode_y, 
                        (ep['end_date'] - ep['start_date']).days,
                        left=ep['start_date'],
                        height=0.8,
                        color=task_colors[ep['task_id']],
                        alpha=0.7,
                        edgecolor='black',
                        linewidth=0.5)
                
                # Add task ID
                mid_date = ep['start_date'] + (ep['end_date'] - ep['start_date']) / 2
                ax2.text(mid_date, episode_y, str(ep['task_id']), 
                        ha='center', va='center', fontsize=8, fontweight='bold')
                
                episode_y += 1
    
    ax2.set_ylim(-1, episode_y)
    ax2.set_ylabel('Episode Index')
    ax2.set_xlabel('Date')
    ax2.set_title('Episodes Colored by Task ID')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add split boundaries
    for split_name, split_data in splits.items():
        if len(split_data) > 0:
            ax2.axvline(split_data.index[0], color=colors[split_name], 
                       linestyle='--', alpha=0.5)
            ax2.axvline(split_data.index[-1], color=colors[split_name], 
                       linestyle='--', alpha=0.5)
    
    # Format x-axis
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    # Log figure info
    logger.log_metric('n_unique_tasks', len(task_colors))
    logger.log_metric('total_episodes', episode_y)
    
    return fig


def analyze_task_leakage(episodes: dict, logger):
    """Analyze potential task leakage between splits."""
    all_tasks = {}
    
    # Collect tasks by split
    for split_name, episode_list in episodes.items():
        tasks = set(ep['task_id'] for ep in episode_list)
        all_tasks[split_name] = tasks
    
    # Check for overlaps
    leakage_found = False
    
    if 'train' in all_tasks and 'val' in all_tasks:
        train_val_overlap = all_tasks['train'] & all_tasks['val']
        if train_val_overlap:
            logger.log_warning(f"Task leakage between train and val: {train_val_overlap}")
            leakage_found = True
    
    if 'train' in all_tasks and 'test' in all_tasks:
        train_test_overlap = all_tasks['train'] & all_tasks['test']
        if train_test_overlap:
            logger.log_warning(f"Task leakage between train and test: {train_test_overlap}")
            leakage_found = True
    
    if 'val' in all_tasks and 'test' in all_tasks:
        val_test_overlap = all_tasks['val'] & all_tasks['test']
        if val_test_overlap:
            logger.log_warning(f"Task leakage between val and test: {val_test_overlap}")
            leakage_found = True
    
    if not leakage_found:
        logger.log_observation("No task leakage detected between splits")
    
    return all_tasks


def main():
    """Run temporal split test with comprehensive logging."""
    # Load configuration
    config = load_config()
    
    # Create experiment logger
    logger = create_experiment_logger(
        "temporal_split",
        "Testing temporal split strategy with continuous method"
    )
    
    try:
        # Log configuration
        logger.log_config(config['data'], 'input')
        logger.log_config(config['temporal_split'], 'decision')
        
        # Step 1: Load data
        logger.log_checkpoint("data_loading", {"status": "started"})
        
        loader = DataLoader(
            config['data']['assets'],
            config['data']['start_date'],
            config['data']['end_date']
        )
        
        data = loader.fetch_data()
        validation_report = loader.validate_data()
        
        # Log data summary
        data_summary = {
            'assets': config['data']['assets'],
            'total_days_fetched': len(data),
            'date_range': f"{data.index[0].date()} to {data.index[-1].date()}",
            'validation': validation_report
        }
        logger.log_data_summary(data_summary)
        
        # Log any data issues
        if validation_report['issues']:
            for issue in validation_report['issues']:
                logger.log_warning(f"Data issue: {issue}")
        
        # Clean data
        clean_data = loader.clean_data(config['temporal_split']['handle_missing'])
        logger.log_checkpoint("data_loading", {
            "status": "completed",
            "clean_data_shape": clean_data.shape
        })
        
        # Calculate and log data metrics
        data_metrics = calculate_data_metrics(clean_data)
        logger.log_metrics(data_metrics)
        
        # Step 2: Create temporal splits
        logger.log_checkpoint("splitting", {"status": "started"})
        
        split_config = TemporalSplitConfig(**config['temporal_split'])
        splitter = TemporalSplitter(split_config)
        
        splits = splitter.split_data(clean_data)
        
        # Step 3: Create episodes
        logger.log_checkpoint("episode_creation", {"status": "started"})
        
        all_episodes = splitter.create_all_episodes()
        
        # Log results
        summary = splitter.summarize_splits()
        logger.log_split_results(splits, summary)
        logger.log_episode_results(all_episodes)
        
        # Save summary DataFrame
        logger.save_dataframe(summary, "split_summary")
        
        # Step 4: Analyze task distribution and leakage
        task_distribution = analyze_task_leakage(all_episodes, logger)
        
        # Log task analysis
        for split_name, tasks in task_distribution.items():
            logger.log_metric(f"{split_name}_unique_tasks", len(tasks))
        
        # Step 5: Visualize
        logger.log_checkpoint("visualization", {"status": "started"})
        
        fig = visualize_splits(clean_data, splits, all_episodes, logger)
        logger.save_figure(fig, "temporal_split_visualization")
        
        # Step 6: Calculate split statistics
        split_stats = {}
        for split_name, split_data in splits.items():
            if len(split_data) > 0:
                split_stats[f"{split_name}_percentage"] = len(split_data) / len(clean_data) * 100
                
                # Calculate returns for each split
                if 'Close' in split_data.columns:
                    returns = split_data['Close'].pct_change()
                    split_stats[f"{split_name}_mean_return"] = returns.mean().mean()
                    split_stats[f"{split_name}_volatility"] = returns.std().mean()
        
        logger.log_metrics(split_stats)
        
        # Step 7: Observations and next steps
        logger.log_observation(
            f"Temporal split created {len(all_episodes.get('train', []))} training episodes "
            f"covering {len(task_distribution.get('train', []))} unique tasks"
        )
        
        if config['temporal_split']['buffer_days'] > 0:
            logger.log_observation(
                f"Buffer of {config['temporal_split']['buffer_days']} days "
                "prevents information leakage between splits"
            )
        
        # Suggest next steps based on results
        if len(all_episodes.get('test', [])) < 5:
            logger.log_next_step(
                "Consider reducing episode_length or stride to create more test episodes"
            )
        
        if any(len(episodes) == 0 for episodes in all_episodes.values()):
            logger.log_next_step(
                "Some splits have no episodes - adjust split ratios or episode parameters"
            )
        
        logger.log_next_step("Try different task_boundaries (quarter, month) to test granularity")
        logger.log_next_step("Experiment with different buffer_days to measure impact")
        logger.log_next_step("Test with overlapping episodes (reduce stride)")
        
        # Mark success
        logger.log_checkpoint("experiment", {"status": "completed", "success": True})
        
    except Exception as e:
        logger.log_error("Experiment failed", e)
        logger.log_checkpoint("experiment", {"status": "failed", "error": str(e)})
        raise
    
    finally:
        # Finalize logging
        logger.finalize()
    
    return logger


if __name__ == "__main__":
    logger = main()
    print(f"\nExperiment logs saved to: {logger.experiment_dir}")