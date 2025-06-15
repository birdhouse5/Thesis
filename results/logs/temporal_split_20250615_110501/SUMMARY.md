# Experiment Summary: temporal_split_20250615_110501

**Date**: 20250615_110501
**Duration**: 0.9 seconds

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

## Errors
- Experiment failed
