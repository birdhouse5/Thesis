#!/bin/bash
# 3_install_dependencies.sh - Core dependencies installation

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

print_status "📦 Installing VariBAD dependencies"

# Core ML and data processing
print_status "Installing core ML and data processing packages..."
pip install pandas numpy scikit-learn

# Financial data
print_status "Installing financial data packages..."
pip install yfinance pyarrow

# Reinforcement learning
print_status "Installing RL packages..."
pip install gym

# Visualization and monitoring
print_status "Installing visualization packages..."
pip install matplotlib seaborn plotly

# Optional development packages
print_status "Installing development packages..."
pip install jupyter ipykernel

# Technical analysis (for indicators)
print_status "Installing technical analysis packages..."
pip install TA-Lib 2>/dev/null || {
    print_warning "TA-Lib installation failed (requires system dependencies)"
    print_status "Installing alternative technical analysis package..."
    pip install talib-binary 2>/dev/null || {
        print_warning "talib-binary also failed. Will use custom implementations."
    }
}

# Install project in development mode
if [ -f "setup.py" ]; then
    print_status "Installing project in development mode..."
    pip install -e .
    print_success "Project installed in development mode"
else
    print_warning "setup.py not found. Skipping project installation."
fi

# Verify key imports
print_status "Verifying installations..."
python -c "
import pandas as pd
import numpy as np
import torch
import sklearn
import matplotlib.pyplot as plt
import yfinance as yf
import gym
print('✅ All core imports successful')
" 2>/dev/null && print_success "All dependencies verified!" || print_warning "Some imports failed - check installation logs"

print_success "Dependencies installation complete!"

# Show installed packages
print_status "Key packages installed:"
pip list | grep -E "(torch|pandas|numpy|matplotlib|yfinance|gym)" || true