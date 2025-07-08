@echo off
echo ========================================
echo Restructuring sp500_loader to Option 2
echo ========================================

REM Check if we're in the right directory
if not exist "sp500_loader.py" (
    echo Error: sp500_loader.py not found!
    echo Please run this script from the directory containing sp500_loader.py
    pause
    exit /b 1
)

echo Creating new directory structure...

REM Create new directories
mkdir core 2>nul
mkdir utils 2>nul
mkdir data 2>nul

echo Moving files...

REM Move the main loader file
move sp500_loader.py core\loader.py

echo Creating __init__.py files...

REM Create main __init__.py
echo # sp500_loader package> __init__.py
echo """>> __init__.py
echo S&P 500 data loader and preprocessing utilities.>> __init__.py
echo """>> __init__.py
echo.>> __init__.py
echo from .core.loader import load_dataset, load_sp500_history, get_ticker_data, get_active_tickers, to_numpy_3d>> __init__.py
echo from .core.splitting import create_quick_loader, QuickSplitLoader>> __init__.py
echo.>> __init__.py
echo __version__ = "1.0.0">> __init__.py
echo __author__ = "Your Name">> __init__.py

REM Create core/__init__.py
echo # Core functionality> core\__init__.py
echo from .loader import *>> core\__init__.py
echo from .splitting import *>> core\__init__.py

REM Create utils/__init__.py
echo # Utility functions> utils\__init__.py
echo """>> utils\__init__.py
echo Utility functions for data validation and analysis.>> utils\__init__.py
echo """>> utils\__init__.py

REM Create splitting.py in core directory
echo Creating splitting.py...
echo import pandas as pd> core\splitting.py
echo import numpy as np>> core\splitting.py
echo from datetime import datetime>> core\splitting.py
echo import random>> core\splitting.py
echo.>> core\splitting.py
echo.>> core\splitting.py
echo def prepare_data_for_splitting(panel_df):>> core\splitting.py
echo     """>> core\splitting.py
echo     Convert multi-index panel data to the format expected by splitting functions.>> core\splitting.py
echo     >> core\splitting.py
echo     Parameters:>> core\splitting.py
echo     ----------->> core\splitting.py
echo     panel_df : pd.DataFrame>> core\splitting.py
echo         Multi-index DataFrame from sp500_loader with (date, ticker) index>> core\splitting.py
echo         >> core\splitting.py
echo     Returns:>> core\splitting.py
echo     -------->> core\splitting.py
echo     tuple: (price_data, is_active_data) as expected by create_temporal_splits>> core\splitting.py
echo     """>> core\splitting.py
echo     >> core\splitting.py
echo     # Extract price data and pivot to get dates as index, tickers as columns>> core\splitting.py
echo     price_data = panel_df['adj_close'].unstack(level='ticker')>> core\splitting.py
echo     >> core\splitting.py
echo     # Extract is_active data and pivot similarly>> core\splitting.py
echo     is_active_data = panel_df['is_active'].unstack(level='ticker').astype(bool)>> core\splitting.py
echo     >> core\splitting.py
echo     print(f"Prepared data shape: {price_data.shape}")>> core\splitting.py
echo     print(f"Date range: {price_data.index[0]} to {price_data.index[-1]}")>> core\splitting.py
echo     print(f"Number of tickers: {price_data.columns.nunique()}")>> core\splitting.py
echo     >> core\splitting.py
echo     return price_data, is_active_data>> core\splitting.py

echo.
echo NOTE: The splitting.py file has been created with basic structure.
echo You'll need to copy the full implementation from the runtime_splitting_workflow artifact.

REM Create a basic validation utility
echo Creating basic validation utility...
echo import pandas as pd> utils\validation.py
echo import numpy as np>> utils\validation.py
echo.>> utils\validation.py
echo.>> utils\validation.py
echo def validate_panel_data(panel_df):>> utils\validation.py
echo     """>> utils\validation.py
echo     Validate the structure and quality of panel data.>> utils\validation.py
echo     >> utils\validation.py
echo     Parameters:>> utils\validation.py
echo     ----------->> utils\validation.py
echo     panel_df : pd.DataFrame>> utils\validation.py
echo         Multi-index DataFrame from sp500_loader>> utils\validation.py
echo     >> utils\validation.py
echo     Returns:>> utils\validation.py
echo     -------->> utils\validation.py
echo     dict: Validation results and statistics>> utils\validation.py
echo     """>> utils\validation.py
echo     >> utils\validation.py
echo     results = {}>> utils\validation.py
echo     >> utils\validation.py
echo     # Check index structure>> utils\validation.py
echo     if not isinstance(panel_df.index, pd.MultiIndex):>> utils\validation.py
echo         results['error'] = "Expected MultiIndex, got single index">> utils\validation.py
echo         return results>> utils\validation.py
echo     >> utils\validation.py
echo     # Check required columns>> utils\validation.py
echo     required_cols = ['adj_close', 'volume', 'is_active']>> utils\validation.py
echo     missing_cols = [col for col in required_cols if col not in panel_df.columns]>> utils\validation.py
echo     if missing_cols:>> utils\validation.py
echo         results['warning'] = f"Missing columns: {missing_cols}">> utils\validation.py
echo     >> utils\validation.py
echo     # Basic statistics>> utils\validation.py
echo     results['shape'] = panel_df.shape>> utils\validation.py
echo     results['date_range'] = (panel_df.index.get_level_values('date').min(), >> utils\validation.py
echo                               panel_df.index.get_level_values('date').max())>> utils\validation.py
echo     results['num_tickers'] = panel_df.index.get_level_values('ticker').nunique()>> utils\validation.py
echo     results['completeness'] = panel_df['adj_close'].count() / len(panel_df)>> utils\validation.py
echo     >> utils\validation.py
echo     return results>> utils\validation.py

REM Create README.md
echo Creating README.md...
echo # S&P 500 Data Loader> README.md
echo.>> README.md
echo A comprehensive toolkit for loading, preprocessing, and splitting S&P 500 historical data.>> README.md
echo.>> README.md
echo ## Installation>> README.md
echo.>> README.md
echo ```python>> README.md
echo # Add the sp500_loader directory to your Python path>> README.md
echo import sys>> README.md
echo sys.path.append('/path/to/sp500_loader')>> README.md
echo ```>> README.md
echo.>> README.md
echo ## Quick Start>> README.md
echo.>> README.md
echo ```python>> README.md
echo from sp500_loader import load_dataset, create_quick_loader>> README.md
echo.>> README.md
echo # Load data>> README.md
echo panel_df = load_dataset('data/sp500_dataset.parquet')>> README.md
echo.>> README.md
echo # Create splits for training>> README.md
echo loader = create_quick_loader(panel_df, episode_length=30)>> README.md
echo.>> README.md
echo # Use in training loop>> README.md
echo for batch in loader.get_episode_batch('train', batch_size=32):>> README.md
echo     # Your training code here>> README.md
echo     pass>> README.md
echo ```>> README.md
echo.>> README.md
echo ## Structure>> README.md
echo.>> README.md
echo - `core/loader.py`: Data loading and S&P 500 historical data download>> README.md
echo - `core/splitting.py`: Train/validation/test splitting and episode creation>> README.md
echo - `utils/validation.py`: Data validation utilities>> README.md
echo - `data/`: Default location for datasets>> README.md

echo.
echo ========================================
echo Restructuring complete!
echo ========================================
echo.
echo Directory structure created:
echo   sp500_loader/
echo   ├── __init__.py
echo   ├── core/
echo   │   ├── __init__.py
echo   │   ├── loader.py (moved from sp500_loader.py)
echo   │   └── splitting.py (basic structure)
echo   ├── utils/
echo   │   ├── __init__.py
echo   │   └── validation.py
echo   ├── data/ (empty, ready for datasets)
echo   └── README.md
echo.
echo IMPORTANT: You need to manually copy the full splitting implementation
echo from the runtime_splitting_workflow artifact into core/splitting.py
echo.
echo The package is now ready to use with:
echo   from sp500_loader import load_dataset, create_quick_loader
echo.
pause