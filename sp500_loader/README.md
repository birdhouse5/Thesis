

## Installation

```python
# Add the sp500_loader directory to your Python path
import sys
sys.path.append('/path/to/sp500_loader')
```

## Quick Start

```python
from sp500_loader import load_dataset, create_quick_loader

# Load data
panel_df = load_dataset('data/sp500_dataset.parquet')

# Create splits for training
loader = create_quick_loader(panel_df, episode_length=30)

# Use in training loop
for batch in loader.get_episode_batch('train', batch_size=32):
    # Your training code here
    pass
```

## Structure

- `core/splitting.py`: Train/validation/test splitting and episode creation
- `utils/validation.py`: Data validation utilities
- `data/`: Default location for datasets
