# Experiment Summary: temporal_split_20250615_110038

**Date**: 20250615_110038
**Duration**: 1.2 seconds

## Description
Testing temporal split strategy with continuous method

## Configuration
### Input Variables
- **assets**: ['SPY', 'QQQ', 'TLT', 'GLD', 'VNQ']
- **start_date**: 2018-01-01
- **end_date**: 2023-12-31

### Decision Variables
- **train_ratio**: 0.7
- **val_ratio**: 0.15
- **test_ratio**: 0.15
- **split_method**: continuous
- **buffer_days**: 20
- **episode_length**: 252
- **min_history**: 60
- **stride**: 21
- **min_common_dates**: 0.95
- **handle_missing**: forward_fill
- **task_boundaries**: year

## Results
- **n_days**: 1577
- **n_features**: 25
- **date_range**: 2017-09-25 to 2023-12-29
- **missing_values**: 0
- **missing_percentage**: 0.0
- **mean_daily_return**: 0.00039520501145618163
- **mean_daily_volatility**: 0.012375667017203903
- **max_daily_return**: 0.09060303682526039
- **min_daily_return**: -0.177277339446962
- **mean_correlation**: 0.2608366639328958
- **max_correlation**: 0.9313473525189891

## Errors
- Experiment failed
