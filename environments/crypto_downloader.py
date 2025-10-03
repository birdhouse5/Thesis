import logging
import sys
from data import PortfolioDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Test the crypto download
logger.info("Testing crypto dataset creation...")

dataset = PortfolioDataset(
    asset_class="crypto",
    data_path="environments/data/crypto_rl_ready_cleaned.parquet",
    proportional=True,  # ADD THIS
    proportions=(0.7, 0.2, 0.1),  # 70% train, 20% val, 10% test
    force_recreate=False  # Change to False since data already exists
)

logger.info(f"Dataset created: {dataset.full_data.shape}")
logger.info(f"Date range: {dataset.full_data['date'].min()} to {dataset.full_data['date'].max()}")
logger.info(f"\nSplit info:")
for split_name in ['train', 'val', 'test']:
    split = dataset.get_split(split_name)
    info = split.get_split_info()
    logger.info(f"  {split_name}: {info['num_days']} days, {info['date_range']}")