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
    force_recreate=True
)

logger.info(f"Dataset created: {dataset.full_data.shape}")
logger.info(f"Date range: {dataset.full_data['date'].min()} to {dataset.full_data['date'].max()}")