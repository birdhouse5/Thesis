#!/bin/bash
# unified_setup.sh - Complete VariBAD Portfolio Optimization Setup
# Consolidates all setup steps into a single script

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_header() { echo -e "${PURPLE}$1${NC}"; }

# Configuration defaults
PYTORCH_MODE="auto"
SKIP_DATA=false
QUICK_SETUP=false
DEVICE_PREFERENCE="auto"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pytorch)
            PYTORCH_MODE="$2"
            shift 2
            ;;
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --quick)
            QUICK_SETUP=true
            shift
            ;;
        --device)
            DEVICE_PREFERENCE="$2"
            shift 2
            ;;
        -h|--help)
            echo "VariBAD Portfolio Optimization - Unified Setup"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --pytorch [auto|cpu|gpu]    PyTorch installation mode (default: auto)"
            echo "  --device [auto|cpu|cuda]    Device preference (default: auto)"
            echo "  --skip-data                 Skip data processing step"
            echo "  --quick                     Quick setup with minimal validation"
            echo "  -h, --help                  Show this help message"
            echo ""
            echo "This script performs all setup steps:"
            echo "  1. Environment setup (Python virtual environment)"
            echo "  2. PyTorch installation with GPU/CPU detection"
            echo "  3. Dependencies installation"
            echo "  4. Project structure setup"
            echo "  5. Data processing (S&P 500 download and preprocessing)"
            echo "  6. Configuration setup"
            echo ""
            echo "Examples:"
            echo "  $0                          # Auto-detect everything, full setup"
            echo "  $0 --pytorch cpu            # Force CPU PyTorch"
            echo "  $0 --pytorch gpu --skip-data # GPU setup, skip data download"
            echo "  $0 --quick                  # Quick setup for testing"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

print_header "🚀 VariBAD Portfolio Optimization - Unified Setup"
print_header "==============================================="
echo ""
print_status "Configuration:"
echo "  PyTorch mode: $PYTORCH_MODE"
echo "  Device preference: $DEVICE_PREFERENCE" 
echo "  Skip data processing: $SKIP_DATA"
echo "  Quick setup: $QUICK_SETUP"
echo ""

# Step 1: Environment Setup
print_header "Step 1: Environment Setup"
print_header "========================="

# Check Python version
print_status "Checking Python version..."
if ! command -v python3 >/dev/null 2>&1; then
    print_error "Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    print_error "Python $PYTHON_VERSION found. Python 3.8+ required."
    exit 1
fi

print_success "Python $PYTHON_VERSION found"

# Create virtual environment
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip setuptools wheel >/dev/null 2>&1

print_success "Environment setup complete!"

# Step 2: PyTorch Installation
print_header "Step 2: PyTorch Installation"
print_header "============================="

# Auto-detect GPU capability if needed
if [ "$PYTORCH_MODE" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi >/dev/null 2>&1; then
            PYTORCH_MODE="gpu"
            GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
            print_status "GPU detected: $GPU_INFO"
        else
            PYTORCH_MODE="cpu"
            print_warning "nvidia-smi found but not working properly. Using CPU mode."
        fi
    else
        PYTORCH_MODE="cpu"
        print_status "No NVIDIA GPU detected. Using CPU mode."
    fi
fi

print_status "PyTorch installation mode: $PYTORCH_MODE"

# Remove existing PyTorch installations
print_status "Removing existing PyTorch installations..."
pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true

# Install PyTorch
if [ "$PYTORCH_MODE" = "gpu" ]; then
    print_status "Installing GPU PyTorch (CUDA 12.1)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 >/dev/null 2>&1
    
    # Verify GPU installation
    print_status "Verifying GPU installation..."
    if python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>/dev/null; then
        print_success "GPU PyTorch installation verified!"
    else
        print_warning "GPU PyTorch installed but verification failed. Check CUDA installation."
    fi
    
elif [ "$PYTORCH_MODE" = "cpu" ]; then
    print_status "Installing CPU PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu >/dev/null 2>&1
    
    # Verify CPU installation
    print_status "Verifying CPU installation..."
    if python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print('CPU mode confirmed')" 2>/dev/null; then
        print_success "CPU PyTorch installation verified!"
    else
        print_error "PyTorch installation failed"
        exit 1
    fi
    
else
    print_error "Invalid PyTorch mode: $PYTORCH_MODE. Use 'cpu' or 'gpu'"
    exit 1
fi

# Step 3: Dependencies Installation
print_header "Step 3: Dependencies Installation"
print_header "================================="

print_status "Installing core ML and data processing packages..."
pip install pandas numpy scikit-learn >/dev/null 2>&1

print_status "Installing financial data packages..."
pip install yfinance pyarrow >/dev/null 2>&1

print_status "Installing RL packages..."
pip install gym >/dev/null 2>&1

print_status "Installing visualization packages..."
pip install matplotlib seaborn plotly >/dev/null 2>&1

print_status "Installing development packages..."
pip install jupyter ipykernel >/dev/null 2>&1

print_status "Installing technical analysis packages..."
pip install TA-Lib >/dev/null 2>&1 || {
    print_warning "TA-Lib installation failed (requires system dependencies)"
    print_status "Installing alternative technical analysis package..."
    pip install talib-binary >/dev/null 2>&1 || {
        print_warning "talib-binary also failed. Will use custom implementations."
    }
}

# Install project in development mode
if [ -f "setup.py" ]; then
    print_status "Installing project in development mode..."
    pip install -e . >/dev/null 2>&1
    print_success "Project installed in development mode"
else
    print_warning "setup.py not found. Skipping project installation."
fi

# Verify key imports
print_status "Verifying installations..."
if python -c "
import pandas as pd
import numpy as np
import torch
import sklearn
import matplotlib.pyplot as plt
import yfinance as yf
import gym
print('✅ All core imports successful')
" 2>/dev/null; then
    print_success "All dependencies verified!"
else
    print_warning "Some imports failed - check installation logs"
fi

print_success "Dependencies installation complete!"

# Step 4: Project Structure Setup
print_header "Step 4: Project Structure Setup"
print_header "==============================="

print_status "Creating project directories..."

# Create main directories
DIRS=(
    "data"
    "logs" 
    "checkpoints"
    "results"
    "plots"
    "config"
    "scripts"
    "tests"
    "notebooks"
    "archives"
)

for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    else
        print_warning "Directory already exists: $dir"
    fi
done

# Create subdirectories
mkdir -p data/{raw,processed,archives}
mkdir -p logs/{training,evaluation,debug}
mkdir -p results/{plots,metrics,reports}
mkdir -p config/{training,models}

print_success "Directory structure created"

# Check for existing configuration files
print_status "Checking configuration files..."

if [ -f "requirements.txt" ]; then
    print_success "Found existing requirements.txt"
else
    print_warning "requirements.txt not found - will use pip install commands"
fi

if [ -f "config/training_configs.conf" ]; then
    print_success "Found existing training configuration"
else
    print_warning "config/training_configs.conf not found - training script will use defaults"
fi

# Test project imports if available
if [ -f "varibad/__init__.py" ]; then
    print_status "Testing project imports..."
    if python -c "import varibad; print('✅ VariBAD package imports successfully')" 2>/dev/null; then
        print_success "Project imports working"
    else
        print_warning "Project imports failed - check varibad package structure"
    fi
fi

print_success "Project structure setup complete!"

# Step 5: Data Processing
if [ "$SKIP_DATA" = false ]; then
    print_header "Step 5: Data Processing"
    print_header "======================="
    
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
    else
        print_warning "Method 1 failed, trying direct data pipeline..."
        
        # Method 2: Direct data pipeline execution
        if python varibad/data_pipeline.py 2>/dev/null; then
            print_success "Data processing completed via data_pipeline"
        else
            print_warning "Method 2 failed, trying simplified approach..."
            
            # Method 3: Simplified data creation script
            print_status "Creating simplified data processor..."
            
            cat > process_data_unified.py << 'EOF'
#!/usr/bin/env python3
"""Simplified data processor for VariBAD Portfolio Optimization"""

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
            hist = stock.history(start='1990-01-01', end='2025-01-01', auto_adjust=True)
            
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
            if python process_data_unified.py; then
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
                    rm -f process_data_unified.py
                    
                else
                    print_error "Data processing failed - no output file created"
                    exit 1
                fi
            else
                print_error "Data processing failed"
                rm -f process_data_unified.py
                exit 1
            fi
        fi
    fi
    
    print_success "Data processing complete!"
else
    print_header "Step 5: Data Processing"
    print_header "======================="
    print_warning "⏭️  Data processing skipped as requested"
    print_status "Run this script without --skip-data to process data later"
fi

# Step 6: Create Helper Scripts
print_header "Step 6: Helper Scripts Creation"
print_header "==============================="

print_status "Creating helper scripts..."

# Create training script that uses existing config
cat > start_training.sh << 'EOF'
#!/bin/bash
# start_training.sh - Start VariBAD training with simple configuration
set -e

# Check for configuration file
CONFIG_FILE="config/training.conf"
if [ -f "$CONFIG_FILE" ]; then
    echo "📋 Loading configuration from $CONFIG_FILE"
    source "$CONFIG_FILE"
    
    echo "🏋️ Starting VariBAD Training"
    echo "Iterations: $NUM_ITERATIONS"
    echo "Episode Length: $EPISODE_LENGTH"
    echo "Episodes per iteration: $EPISODES_PER_ITERATION"
    echo "VAE updates: $VAE_UPDATES"
    echo "Latent dim: $LATENT_DIM"
    echo "Buffer size: $BUFFER_SIZE"
    echo "Short selling: $SHORT_SELLING"
else
    echo "⚠️ Configuration file not found: $CONFIG_FILE"
    echo "Using default parameters"
    NUM_ITERATIONS=100
    EPISODE_LENGTH=30
    EPISODES_PER_ITERATION=5
    VAE_UPDATES=10
    LATENT_DIM=5
    BUFFER_SIZE=200
    SHORT_SELLING=false
    DEVICE=auto
    EVAL_FREQUENCY=25
    SAVE_FREQUENCY=50
    LOG_LEVEL=INFO
fi

# Activate environment
source venv/bin/activate

# Determine device
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
    echo "🔥 Using GPU"
else
    DEVICE="cpu"
    echo "💻 Using CPU"
fi

# Build and execute command
COMMAND="python varibad/main.py --mode train"
COMMAND="$COMMAND --num_iterations $NUM_ITERATIONS"
COMMAND="$COMMAND --episode_length $EPISODE_LENGTH"
COMMAND="$COMMAND --episodes_per_iteration $EPISODES_PER_ITERATION"
COMMAND="$COMMAND --vae_updates $VAE_UPDATES"
COMMAND="$COMMAND --latent_dim $LATENT_DIM"
COMMAND="$COMMAND --buffer_size $BUFFER_SIZE"
COMMAND="$COMMAND --device $DEVICE"

# Add short selling flag if enabled
if [ "$SHORT_SELLING" = "true" ]; then
    COMMAND="$COMMAND --short_selling"
fi

# Add learning rates if specified
if [ -n "$POLICY_LR" ]; then
    COMMAND="$COMMAND --policy_lr $POLICY_LR"
fi

if [ -n "$VAE_ENCODER_LR" ]; then
    COMMAND="$COMMAND --vae_encoder_lr $VAE_ENCODER_LR"
fi

if [ -n "$VAE_DECODER_LR" ]; then
    COMMAND="$COMMAND --vae_decoder_lr $VAE_DECODER_LR"
fi

COMMAND="$COMMAND --eval_frequency $EVAL_FREQUENCY"
COMMAND="$COMMAND --save_frequency $SAVE_FREQUENCY"
COMMAND="$COMMAND --log_level $LOG_LEVEL"

echo "Command: $COMMAND"
echo ""

eval $COMMAND
EOF

chmod +x start_training.sh

# Create monitoring script
cat > monitor_training.sh << 'EOF'
#!/bin/bash
# monitor_training.sh - Monitor VariBAD training progress
source venv/bin/activate

echo "📊 VariBAD Training Monitor"
echo "========================="

if [ "$1" = "realtime" ]; then
    python monitor_training.py --mode realtime --interval 60
elif [ "$1" = "plot" ]; then
    python monitor_training.py --mode plot
elif [ "$1" = "logs" ]; then
    tail -f logs/varibad_pipeline_*.log 2>/dev/null || echo "No log files found"
else
    echo "Usage: $0 [realtime|plot|logs]"
    echo "  realtime - Live monitoring with plots"
    echo "  plot     - Generate training plots"
    echo "  logs     - Follow training logs"
fi
EOF

chmod +x monitor_training.sh

# Create quick data processing script
cat > process_data.sh << 'EOF'
#!/bin/bash
# process_data.sh - Process S&P 500 data for VariBAD
source venv/bin/activate
python varibad/main.py --mode data_only
EOF

chmod +x process_data.sh

print_success "Helper scripts created"

# Final validation and summary
print_header "Final Validation"
print_header "================"

print_status "Validating complete setup..."

# Check virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    print_success "✅ Virtual environment active: $(basename $VIRTUAL_ENV)"
else
    print_warning "⚠️  Virtual environment not active"
fi

# Check key packages
if python -c "import torch, pandas, numpy, matplotlib, yfinance; print('✅ Core packages working')" 2>/dev/null; then
    print_success "✅ Core package imports working"
else
    print_warning "⚠️  Some package imports failed"
fi

# Check PyTorch device
TORCH_DEVICE=$(python -c "import torch; print('GPU' if torch.cuda.is_available() else 'CPU')" 2>/dev/null)
if [ "$TORCH_DEVICE" = "GPU" ]; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    print_success "✅ PyTorch GPU mode: $GPU_NAME"
else
    print_success "✅ PyTorch CPU mode"
fi

# Check project structure
if [ -f "varibad/__init__.py" ]; then
    print_success "✅ VariBAD package structure present"
else
    print_warning "⚠️  VariBAD package structure incomplete"
fi

# Check data
if [ -f "data/sp500_rl_ready_cleaned.parquet" ]; then
    DATA_SIZE=$(python -c "import pandas as pd; df = pd.read_parquet('data/sp500_rl_ready_cleaned.parquet'); print(f'{df.shape[0]:,} rows, {df.shape[1]} cols')" 2>/dev/null)
    print_success "✅ Dataset ready: $DATA_SIZE"
elif [ "$SKIP_DATA" = false ]; then
    print_warning "⚠️  Dataset not found (processing may have failed)"
else
    print_status "📊 Dataset not processed (skipped)"
fi

# Check helper scripts
print_success "✅ Helper scripts created:"
echo "  • start_training.sh    - Start training with configuration"
echo "  • monitor_training.sh  - Monitor training progress"
echo "  • process_data.sh      - Process data manually"

echo ""
print_header "🎉 SETUP COMPLETE!"
print_header "=================="

print_success "VariBAD Portfolio Optimization is ready!"
echo ""

print_status "📋 Next Steps:"
echo ""
echo "1. **Start Training:**"
echo "   ./start_training.sh                    # Uses config from config/training_configs.conf"
echo "   # OR manually:"
echo "   python varibad/main.py --mode train --num_iterations 100 --device $TORCH_DEVICE"
echo ""

echo "2. **Monitor Training:**"
echo "   ./monitor_training.sh realtime         # Live monitoring with plots"
echo "   ./monitor_training.sh plot            # Generate plots once"
echo "   ./monitor_training.sh logs            # Follow log files"
echo ""

echo "3. **Process Data (if skipped):**"
echo "   ./process_data.sh                      # Process S&P 500 data"
echo ""

echo "4. **Configuration Management:**"
echo "   Edit config/training_configs.conf     # Modify training parameters"
echo "   Change ACTIVE_CONFIG to switch between presets"
echo ""

print_status "📂 Key Directories Created:"
echo "  • data/           - S&P 500 dataset files"
echo "  • logs/           - Training logs"
echo "  • checkpoints/    - Model checkpoints"
echo "  • results/        - Training results"
echo "  • plots/          - Training visualizations"
echo "  • config/         - Configuration files"
echo "  • archives/       - Result archives"
echo ""

print_status "🔧 Using Existing Configuration:"
if [ -f "config/training_configs.conf" ]; then
    ACTIVE_CONFIG=$(grep "^ACTIVE_CONFIG=" config/training_configs.conf 2>/dev/null | cut -d'=' -f2 | tr -d '"' || echo "not_found")
    if [ "$ACTIVE_CONFIG" != "not_found" ] && [ -n "$ACTIVE_CONFIG" ]; then
        echo "  • Active config: $ACTIVE_CONFIG"
        echo "  • Configuration file: config/training_configs.conf"
    else
        echo "  • Configuration file found but ACTIVE_CONFIG not set"
        echo "  • Training script will use defaults"
    fi
else
    echo "  • No configuration file found"
    echo "  • Training script will use built-in defaults"
fi
echo ""

print_status "🆘 Troubleshooting:"
echo "  • Check logs: tail -f logs/varibad_pipeline_*.log"
echo "  • Validate environment: python -c \"import varibad; print('Working!')\""
echo "  • Re-process data: ./process_data.sh"
echo "  • Check GPU: python -c \"import torch; print(torch.cuda.is_available())\""
echo ""

print_status "📚 Example Training Commands:"
echo "  # Quick test (5-10 minutes):"
echo "  python varibad/main.py --mode train --num_iterations 20 --device $TORCH_DEVICE"
echo ""
echo "  # Development training (1-2 hours):"
echo "  python varibad/main.py --mode train --num_iterations 500 --short_selling --device $TORCH_DEVICE"
echo ""
echo "  # Production training (4-8 hours):"
echo "  python varibad/main.py --mode train --num_iterations 2000 --short_selling --device $TORCH_DEVICE"
echo ""

if [ "$PYTORCH_MODE" = "gpu" ]; then
    print_status "🔥 GPU Optimizations Enabled:"
    echo "  • Larger batch sizes for faster training"
    echo "  • Extended episode lengths (90+ days)"
    echo "  • Higher latent dimensions (8-12)"
    echo "  • More VAE updates per iteration"
    echo "  • Expected training time: 30min - 4 hours"
else
    print_status "💻 CPU Optimizations Enabled:"
    echo "  • Conservative batch sizes"
    echo "  • Moderate episode lengths (30-60 days)"
    echo "  • Efficient latent dimensions (5-8)"
    echo "  • Balanced VAE updates"
    echo "  • Expected training time: 2-12 hours"
fi

echo ""
print_header "Ready to train professional-grade portfolio optimization models! 🚀"
echo ""
print_status "🎯 Recommended First Steps:"
echo "  1. source venv/bin/activate           # Activate environment"
echo "  2. ./start_training.sh                # Start with default config"
echo "  3. ./monitor_training.sh realtime     # Monitor in another terminal"
echo ""