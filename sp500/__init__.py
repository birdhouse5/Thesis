# sp500_loader package

from .core.loader import load_dataset, load_sp500_history, get_ticker_data, get_active_tickers, to_numpy_3d
from .core.splitting import create_quick_loader, QuickSplitLoader
from .core.environment import MinimalPortfolioEnv, create_env_from_loader

__version__ = "1.0.0"
__author__ = "Your Name"
