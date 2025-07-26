#!/bin/bash
# setup_all.sh - Complete VariBAD setup in separate steps

set -e

# Color codes
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

# Default options
PYTORCH_MODE="auto"
SKIP_DATA=false
QUICK_SETUP=false

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
        -h|--help)
            echo "VariBAD Portfolio Optimization - Complete Setup"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --pytorch [auto|cpu|gpu]    PyTorch installation mode (default: auto)"
            echo "  --skip-data                 Skip data processing step"
            echo "  --quick                     Quick setup with minimal validation"
            echo "  -h, --help                  Show this help message"
            echo ""
            echo "This script runs all setup steps in sequence:"
            echo "  1. Environment setup"
            echo "  2. PyTorch installation" 
            echo "  3. Dependencies installation"
            echo "  4. Project structure setup"
            echo "  5. Data processing (unless --skip-data)"
            echo ""
            echo "Examples:"
            echo "  $0                          # Auto-detect GPU, full setup"
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

print_header "🚀 VariBAD Portfolio Optimization - Complete Setup"
print_header "=================================================="
echo ""
print_status "Setup configuration:"
echo "  PyTorch mode: $PYTORCH_MODE"
echo "  Skip data processing: $SKIP_DATA"
echo "  Quick setup: $QUICK_SETUP"
echo ""

# Function to run step with error handling
run_step() {
    local step_num="$1"
    local step_name="$2"
    local script_name="$3"
    local step_args="${4:-}"
    
    print_header "Step $step_num: $step_name"
    print_header "$(printf '=%.0s' {1..50})"
    
    if [ -f "$script_name" ]; then
        if [ -n "$step_args" ]; then
            if ./"$script_name" $step_args; then
                print_success "✅ Step $step_num completed: $step_name"
            else
                print_error "❌ Step $step_num failed: $step_name"
                print_status "You can run this step manually with: ./$script_name $step_args"
                exit 1
            fi
        else
            if ./"$script_name"; then
                print_success "✅ Step $step_num completed: $step_name"
            else
                print_error "❌ Step $step_num failed: $step_name"
                print_status "You can run this step manually with: ./$script_name"
                exit 1
            fi
        fi
    else
        print_error "❌ Script not found: $script_name"
        print_status "Make sure all setup scripts are in the current directory"
        exit 1
    fi
    
    echo ""
    sleep 1
}

# Check if all required scripts exist
REQUIRED_SCRIPTS=(
    "1_setup_environment.sh"
    "2_install_pytorch.sh"
    "3_install_dependencies.sh"
    "4_setup_project_structure.sh"
    "5_process_data.sh"
)

print_status "Checking for required setup scripts..."
for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        print_error "Required script not found: $script"
        print_status "Make sure all setup scripts are in the current directory"
        exit 1
    fi
    chmod +x "$script"
done
print_success "All required scripts found"
echo ""

# Confirm setup
if [ "$QUICK_SETUP" = false ]; then
    print_warning "This will run the complete VariBAD setup process."
    print_status "Steps to be executed:"
    echo "  1. Environment setup (Python virtual environment)"
    echo "  2. PyTorch installation ($PYTORCH_MODE mode)"
    echo "  3. Dependencies installation (pandas, numpy, etc.)"
    echo "  4. Project structure setup (directories, configs)"
    if [ "$SKIP_DATA" = false ]; then
        echo "  5. Data processing (S&P 500 download and preprocessing)"
    else
        echo "  5. Data processing (SKIPPED)"
    fi
    echo ""
    
    read -p "Continue with setup? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Setup cancelled"
        exit 0
    fi
fi

echo ""
print_header "🔄 Starting VariBAD Setup Process"
print_header "================================="
echo ""

# Step 1: Environment Setup
run_step "1" "Environment Setup" "1_setup_environment.sh"

# Step 2: PyTorch Installation
run_step "2" "PyTorch Installation" "2_install_pytorch.sh" "$PYTORCH_MODE"

# Step 3: Dependencies Installation
run_step "3" "Dependencies Installation" "3_install_dependencies.sh"

# Step 4: Project Structure Setup
run_step "4" "Project Structure Setup" "4_setup_project_structure.sh"

# Step 5: Data Processing (optional)
if [ "$SKIP_DATA" = false ]; then
    run_step "5" "Data Processing" "5_process_data.sh"
else
    print_header "Step 5: Data Processing"
    print_header "$(printf '=%.0s' {1..50})"
    print_warning "⏭️  Data processing skipped as requested"
    print_status "Run './5_process_data.sh' manually when ready"
    echo ""
fi

# Final validation
print_header "🎯 Setup Validation"
print_header "=================="

print_status "Validating installation..."

# Check virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    print_success "✅ Virtual environment active: $(basename $VIRTUAL_ENV)"
else
    print_warning "⚠️  Virtual environment not active"
    print_status "Activate with: source venv/bin/activate"
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

echo ""
print_header "🎉 SETUP COMPLETE!"
print_header "=================="

print_success "VariBAD Portfolio Optimization is ready!"
echo ""

print_status "📋 Next Steps:"
echo ""
echo "1. **Activate Environment** (if not already active):"
echo "   source venv/bin/activate"
echo ""

if [ "$SKIP_DATA" = true ]; then
    echo "2. **Process Data** (you skipped this step):"
    echo "   ./5_process_data.sh"
    echo ""
    echo "3. **Start Training** (after data processing):"
else
    echo "2. **Start Training:**"
fi

echo "   ./6_train_model.sh --mode quick    # 5-10 minute test"
echo "   ./6_train_model.sh --mode dev      # 1-2 hour development"
echo "   ./6_train_model.sh --mode full     # 6-12 hour production"
echo ""

echo "3. **Monitor Training:**"
echo "   ./7_monitor_training.sh            # Interactive monitor"
echo "   python monitor_training.py --mode realtime"
echo ""

echo "4. **Alternative Training Commands:**"
echo "   python varibad/main.py --mode train --num_iterations 10 --device $TORCH_DEVICE"
echo ""

print_status "📂 Key Directories Created:"
echo "  • data/           - S&P 500 dataset files"
echo "  • logs/           - Training logs"
echo "  • checkpoints/    - Model checkpoints"
echo "  • results/        - Training results"
echo "  • plots/          - Training visualizations"
echo ""

print_status "🔧 Configuration Files:"
echo "  • requirements.txt    - Python dependencies"
echo "  • .gitignore         - Git ignore rules"
echo "  • README.md          - Project documentation"
echo ""

print_status "🆘 If Issues Occur:"
echo "  • Check logs in: logs/"
echo "  • Re-run individual steps: ./[step_number]_*.sh"
echo "  • Validate environment: python -c \"import varibad; print('Working!')\""
echo ""

print_header "Ready to train professional-grade portfolio optimization models! 🚀"