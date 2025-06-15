# Experiment Summary: temporal_split_20250615_111839

**Date**: 20250615_111839
**Duration**: 2.9 seconds

## Description
Testing temporal split strategy with continuous method

## Configuration
### Input Variables
- **assets**: ['SPY', 'QQQ', 'TLT', 'GLD', 'VNQ']
- **start_date**: 2018-01-01
- **end_date**: 2023-12-31
- **min_common_dates**: 0.95
- **handle_missing**: forward_fill

### Decision Variables
- **train_ratio**: 0.7
- **val_ratio**: 0.15
- **test_ratio**: 0.15
- **split_method**: continuous
- **buffer_days**: 20
- **episode_length**: 252
- **min_history**: 60
- **stride**: 21
- **task_boundaries**: year

## Results
- **n_days**: 1577
- **n_features**: 25
- **date_range**: 2017-09-25 to 2023-12-29
- **missing_values**: 0
- **missing_percentage**: 0.0
- **mean_daily_return**: 0.00039520501639486624
- **mean_daily_volatility**: 0.012375668120459291
- **max_daily_return**: 0.09060312377302471
- **min_daily_return**: -0.17727730195443525
- **mean_correlation**: 0.2608368800977456
- **max_correlation**: 0.9313470913928565
- **train_unique_tasks**: 5
- **val_unique_tasks**: 0
- **test_unique_tasks**: 0
- **n_unique_tasks**: 5
- **total_episodes**: 38
- **train_percentage**: 69.94292961318959
- **train_mean_return**: 0.0005437752558733798
- **train_volatility**: 0.012119786696192184
- **val_percentage**: 13.696892834495877
- **val_mean_return**: -0.0004342930532288243
- **val_volatility**: 0.014909254056331584
- **test_percentage**: 13.823715916296766
- **test_mean_return**: 0.000629244442603792
- **test_volatility**: 0.010189724052512356

## Observations
- No task leakage detected between splits
- Temporal split created 38 training episodes covering 5 unique tasks
- Buffer of 20 days prevents information leakage between splits

## Next Steps
- Consider reducing episode_length or stride to create more test episodes
- Some splits have no episodes - adjust split ratios or episode parameters
- Try different task_boundaries (quarter, month) to test granularity
- Experiment with different buffer_days to measure impact
- Test with overlapping episodes (reduce stride)
