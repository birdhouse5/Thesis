#!/bin/bash
# VariBAD Portfolio Optimization - Cloud Setup Script
# Run this script on a fresh cloud instance to set up the environment

set -e  # Exit on any error

echo "🚀 Setting up VariBAD Portfolio Optimization environment..."

# Update system
echo "📦 Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    htop \
    tmux \
    build-essential

# Create project directory and virtual environment
echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU version for compatibility, change if you have GPU)
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
echo "📚 Installing Python packages..."
pip install -r requirements.txt

# Verify installation
echo "✅ Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p data logs checkpoints results plots

# Set up permissions
chmod +x main.py

echo "🎉 Setup complete!"
echo ""
echo "🏃 Quick start commands:"
echo "  source venv/bin/activate  # Activate environment"
echo "  python main.py --mode data_only --log_level INFO  # Process data only"
echo "  python main.py --mode train --num_iterations 100  # Train model"
echo ""
echo "💡 For long training runs, use tmux:"
echo "  tmux new-session -d -s varibad 'source venv/bin/activate && python main.py --mode train --num_iterations 1000'"
echo "  tmux attach-session -t varibad  # To monitor"
echo "  tmux detach  # Ctrl+B then D to detach"