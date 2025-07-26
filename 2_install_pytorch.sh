#!/bin/bash
# 2_install_pytorch.sh - PyTorch installation with GPU/CPU detection

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

print_status "🔥 Installing PyTorch for VariBAD Portfolio Optimization"

# Parse command line arguments
MODE="auto"
if [ $# -gt 0 ]; then
    MODE="$1"
fi

# Auto-detect GPU capability
if [ "$MODE" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi >/dev/null 2>&1; then
            MODE="gpu"
            GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
            print_status "GPU detected: $GPU_INFO"
        else
            MODE="cpu"
            print_warning "nvidia-smi found but not working properly. Using CPU mode."
        fi
    else
        MODE="cpu"
        print_status "No NVIDIA GPU detected. Using CPU mode."
    fi
fi

print_status "Installation mode: $MODE"

# Remove existing PyTorch installations
print_status "Removing existing PyTorch installations..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install PyTorch
if [ "$MODE" = "gpu" ]; then
    print_status "Installing GPU PyTorch (CUDA 12.1)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Verify GPU installation
    print_status "Verifying GPU installation..."
    if python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>/dev/null; then
        print_success "GPU PyTorch installation verified!"
    else
        print_warning "GPU PyTorch installed but verification failed. Check CUDA installation."
    fi
    
elif [ "$MODE" = "cpu" ]; then
    print_status "Installing CPU PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # Verify CPU installation
    print_status "Verifying CPU installation..."
    if python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print('CPU mode confirmed')" 2>/dev/null; then
        print_success "CPU PyTorch installation verified!"
    else
        print_error "PyTorch installation failed"
        exit 1
    fi
    
else
    print_error "Invalid mode: $MODE. Use 'cpu' or 'gpu'"
    exit 1
fi

print_success "PyTorch installation complete!"
print_status "Usage: ./2_install_pytorch.sh [cpu|gpu|auto]"