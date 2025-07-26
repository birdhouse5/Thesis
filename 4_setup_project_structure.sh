#!/bin/bash
# 4_setup_project_structure.sh - Create directories and basic project files

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

print_status "📁 Setting up VariBAD project structure"

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

# Create basic configuration files
print_status "Creating basic configuration files..."

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data files
data/*.parquet
data/*.csv
data/*.h5
data/raw/
data/processed/

# Model files
checkpoints/*.pt
checkpoints/*.pth
*.pkl

# Logs
logs/*.log
*.log

# Results
results/
plots/*.png
plots/*.pdf

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# Environment variables
.env
.env.local
EOF
    print_success "Created .gitignore"
else
    print_warning ".gitignore already exists"
fi

# Create requirements.txt
if [ ! -f "requirements.txt" ]; then
    cat > requirements.txt << 'EOF'
# Core ML packages
torch>=2.0.0
numpy>=1.21.0
pandas>=1.5.0
scikit-learn>=1.0.0

# Financial data
yfinance>=0.2.0
pyarrow>=10.0.0

# RL environment
gym>=0.21.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0

# Development
jupyter>=1.0.0
ipykernel>=6.0.0

# Technical analysis (optional)
TA-Lib; sys_platform != "win32"
talib-binary; sys_platform == "win32"
EOF
    print_success "Created requirements.txt"
else
    print_warning "requirements.txt already exists"
fi

# Create basic README if it doesn't exist
if [ ! -f "README.md" ] && [ ! -f "CLAUDE.md" ]; then
    cat > README.md << 'EOF'
# VariBAD Portfolio Optimization

Regime-agnostic reinforcement learning for portfolio optimization using VariBAD (Variational Bayes Adaptive Deep RL).

## Quick Start

1. **Setup Environment**
   ```bash
   ./1_setup_environment.sh
   source venv/bin/activate
   ```

2. **Install PyTorch**
   ```bash
   ./2_install_pytorch.sh auto  # or 'cpu' or 'gpu'
   ```

3. **Install Dependencies**
   ```bash
   ./3_install_dependencies.sh
   ```

4. **Setup Project**
   ```bash
   ./4_setup_project_structure.sh
   ```

5. **Process Data**
   ```bash
   ./5_process_data.sh
   ```

6. **Start Training**
   ```bash
   ./6_train_model.sh
   ```

## Project Structure

- `varibad/` - Core VariBAD implementation
- `data/` - Dataset files
- `logs/` - Training logs
- `checkpoints/` - Model checkpoints
- `results/` - Training results
- `scripts/` - Utility scripts

## Features

- Portfolio optimization for 30 S&P 500 stocks (1990-2025)
- Automatic regime detection without manual labels
- Long/short trading strategies
- Risk-adjusted returns via Differential Sharpe Ratio

## Next Steps

Run the setup scripts in order, then use:
- `python varibad/main.py --mode data_only` to process data
- `python varibad/main.py --mode train` to start training
- `python monitor_training.py --mode realtime` to monitor progress
EOF
    print_success "Created README.md"
fi

# Test project imports
if [ -f "varibad/__init__.py" ]; then
    print_status "Testing project imports..."
    if python -c "import varibad; print('✅ VariBAD package imports successfully')" 2>/dev/null; then
        print_success "Project imports working"
    else
        print_warning "Project imports failed - check varibad package structure"
    fi
fi

print_success "Project structure setup complete!"

# Show final structure
print_status "Project structure:"
tree -L 2 -a 2>/dev/null || {
    print_status "Directory structure (install 'tree' for better view):"
    find . -type d -name ".*" -prune -o -type d -print | head -20
}