#!/bin/bash
# 5_process_data.sh - Download and process S&P 500 data

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if virtual environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    print_error "Virtual environment not active. Run 'source venv/bin/activate' first."
    exit 1
fi

print_status "📊 Processing S&P 500 data for VariBAD training"

# Check if data already exists
FINAL_DATA="data/sp500_rl_ready_cleaned.parquet"

if [ -f "$FINAL_DATA" ]; then
    print_warning "Data already exists at $FINAL_DATA"
    
    # Validate existing data
    if python -c "
import pandas as pd
import numpy as np
try:
    df = pd.read_parquet('$FINAL_DATA')
    print(f'✅ Data validation:')
    print(f'   Shape: {df.shape}')
    print(f'   Date range: {df[\"date\"].min().date()} to {df[\"date\"].max().date()}')
    print(f'   Tickers: {df[\"ticker\"].nunique()}')
    print(f'   Features: {len(df.columns)}')
    print(f'   Missing values: {df.isnull().sum().sum()}')
    
    if df.shape[0] > 10000 and df['ticker'].nunique() >= 25:
        print('Data appears valid!')
        exit(0)
    else:
        print('Data appears incomplete, will regenerate...')
        exit(1)
except Exception as e:
    print(f'Data validation failed: {e}')
    print('Will regenerate data...')
    exit(1)
" 2>/dev/null; then
        print_success "Existing data is valid!"
        exit 0
    else
        print_warning "Existing data invalid, regenerating..."
        rm -f "$FINAL_DATA"
    fi
fi

# Try different approaches to process data
print_status "Attempting to process data..."

# Method 1: Use varibad.main module
if python varibad/main.py --mode data_only 2>/dev/null; then
    print_success "Data processing completed via varibad.main"
    exit 0
fi

print_warning "Method 1 failed, trying direct data pipeline..."

# Method 2: Direct data pipeline execution
if python varibad/data_pipeline.py 2>/dev/null; then
    print_success "Data processing completed via data_pipeline"
    exit 0
fi

print_warning "Method 2 failed, trying simplified approach..."

# Method 3: Simplified data creation script
print_status "Creating simplified data processor..."

cat > process_data_simple.py << 'EOF'
#!/usr/bin/env python3
"""
Simplified data processor for VariBAD Portfolio Optimization
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sp500_data():
    """Create simplified S&P 500 dataset."""
    
    # 30 S&P 500 companies
    tickers = [
        'IBM', 'MSFT', 'ORCL', 'INTC', 'HPQ', 'CSCO',  # Tech
        'JPM', 'BAC', 'WFC', 'C', 'AXP',              # Financial
        'JNJ', 'PFE', 'MRK', 'ABT',                   # Healthcare
        'KO', 'PG', 'WMT', 'PEP',                     # Consumer Staples
        'XOM', 'CVX', 'COP',                          # Energy
        'GE', 'CAT', 'BA',                            # Industrials
        'HD', 'MCD',                                  # Consumer Disc
        'SO', 'D',                                    # Utilities
        'DD'                                          # Materials
    ]
    
    logger.info(f"Downloading data for {len(tickers)} tickers...")
    
    all_data = []
    
    for i, ticker in enumerate(tickers):
        try:
            logger.info(f"Downloading {ticker} ({i+1}/{len(tickers)})...")
            
            # Download data
            stock = yf.Ticker(ticker)
            hist = stock.history(start='2020-01-01', end='2025-01-01', auto_adjust=True)
            
            if hist.empty:
                logger.warning(f"No data found for {ticker}")
                continue
            
            # Reset index and add ticker
            hist = hist.reset_index()
            hist['ticker'] = ticker
            
            # Rename columns
            hist = hist.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Add basic features
            hist['returns'] = hist['close'].pct_change()
            hist['log_returns'] = np.log(hist['close'] / hist['close'].shift(1))
            
            # Simple moving averages
            hist['sma_5'] = hist['close'].rolling(5).mean()
            hist['sma_20'] = hist['close'].rolling(20).mean()
            
            # Simple RSI
            delta = hist['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean() 
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            hist['rsi'] = 100 - (100 / (1 + rs))
            
            # Volatility
            hist['volatility_5d'] = hist['returns'].rolling(5).std()
            hist['volatility_20d'] = hist['returns'].rolling(20).std()
            
            # Normalize key features
            for col in ['sma_5', 'sma_20', 'close']:
                hist[f'{col}_norm'] = (hist[col] - hist[col].min()) / (hist[col].max() - hist[col].min())
            
            hist['rsi_norm'] = (hist['rsi'] - 50) / 50
            
            # Select columns
            hist = hist[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume',
                        'returns', 'log_returns', 'sma_5_norm', 'sma_20_norm', 
                        'close_norm', 'rsi_norm', 'volatility_5d', 'volatility_20d']]
            
            all_data.append(hist)
            
        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No data was successfully downloaded")
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = combined_data.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    # Add market features
    market_data = combined_data.groupby('date')['returns'].mean().reset_index()
    market_data = market_data.rename(columns={'returns': 'market_return'})
    
    combined_data = combined_data.merge(market_data, on='date', how='left')
    combined_data['excess_returns'] = combined_data['returns'] - combined_data['market_return']
    
    # Clean data
    combined_data = combined_data.dropna(subset=['date', 'ticker', 'close', 'returns'])
    
    # Fill remaining NaNs
    numeric_columns = combined_data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col.endswith('_norm'):
            combined_data[col] = combined_data[col].fillna(0.0)
        elif 'rsi' in col.lower():
            combined_data[col] = combined_data[col].fillna(50.0)
        else:
            combined_data[col] = combined_data[col].fillna(method='ffill').fillna(method='bfill')
    
    logger.info(f"Final dataset: {combined_data.shape}")
    logger.info(f"Date range: {combined_data['date'].min().date()} to {combined_data['date'].max().date()}")
    logger.info(f"Tickers: {combined_data['ticker'].nunique()}")
    
    return combined_data

if __name__ == "__main__":
    # Create directories
    os.makedirs("data", exist_ok=True)
    
    # Create dataset
    df = create_sp500_data()
    
    # Save dataset
    output_path = "data/sp500_rl_ready_cleaned.parquet"
    df.to_parquet(output_path)
    
    print(f"✅ Dataset created successfully: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Tickers: {df['ticker'].nunique()}")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
EOF

# Run simplified data processor
print_status "Running simplified data processor..."
if python process_data_simple.py; then
    print_success "Data processing completed!"
    
    # Verify the data
    if [ -f "$FINAL_DATA" ]; then
        print_success "Dataset created at: $FINAL_DATA"
        
        # Show dataset info
        python -c "
import pandas as pd
df = pd.read_parquet('$FINAL_DATA')
print(f'📊 Dataset Summary:')
print(f'   Shape: {df.shape}')
print(f'   Date range: {df[\"date\"].min().date()} to {df[\"date\"].max().date()}')
print(f'   Tickers: {df[\"ticker\"].nunique()}')
print(f'   Features: {len(df.columns)}')
print(f'   Missing values: {df.isnull().sum().sum()}')
print(f'   Sample tickers: {list(df[\"ticker\"].unique()[:5])}...')
"
        
        # Clean up temporary file
        rm -f process_data_simple.py
        
    else
        print_error "Data processing failed - no output file created"
        exit 1
    fi
else
    print_error "Data processing failed"
    rm -f process_data_simple.py
    exit 1
fi

print_success "Data processing complete!"
print_status "Next step: ./6_train_model.sh"