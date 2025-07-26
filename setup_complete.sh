#!/bin/bash
# VariBAD Portfolio Optimization - Simplified Complete Setup Script
# One command to set up everything from scratch (no tmux dependencies)

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

# Default mode
MODE="cpu"
SKIP_SYSTEM_UPDATE=false
QUICK_SETUP=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --skip-system-update)
            SKIP_SYSTEM_UPDATE=true
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
            echo "  --mode [cpu|gpu]           Setup mode (default: cpu)"
            echo "  --skip-system-update       Skip system package updates"
            echo "  --quick                    Quick setup (skip some validation)"
            echo "  -h, --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --mode gpu              # Setup for GPU training"
            echo "  $0 --mode cpu --quick      # Quick CPU setup"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate mode
if [[ "$MODE" != "cpu" && "$MODE" != "gpu" ]]; then
    print_error "Invalid mode: $MODE. Use 'cpu' or 'gpu'"
    exit 1
fi

print_header "🚀 VariBAD Portfolio Optimization - Complete Setup"
print_header "========================================================="
print_status "Setup mode: $MODE"
print_status "Quick setup: $QUICK_SETUP"
echo ""

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            echo "ubuntu"
        elif [ -f /etc/redhat-release ]; then
            echo "centos"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Update system packages
update_system() {
    if [ "$SKIP_SYSTEM_UPDATE" = true ]; then
        print_warning "Skipping system update"
        return
    fi

    print_status "Updating system packages..."
    OS=$(detect_os)
    
    case $OS in
        "ubuntu")
            sudo apt-get update -y >/dev/null 2>&1
            sudo apt-get upgrade -y >/dev/null 2>&1
            sudo apt-get install -y python3 python3-pip python3-venv git wget curl build-essential >/dev/null 2>&1
            ;;
        "centos")
            sudo yum update -y >/dev/null 2>&1
            sudo yum install -y python3 python3-pip git wget curl gcc gcc-c++ make >/dev/null 2>&1
            ;;
        "macos")
            if command -v brew >/dev/null 2>&1; then
                brew update >/dev/null 2>&1
                brew install python3 git wget curl >/dev/null 2>&1
            else
                print_warning "Homebrew not found. Please install manually: https://brew.sh"
            fi
            ;;
        *)
            print_warning "Unknown OS. Please install Python 3.8+, pip, git, and build tools manually."
            ;;
    esac
    
    print_success "System packages updated"
}

# Check Python version
check_python() {
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
}

# Check GPU availability
check_gpu() {
    if [ "$MODE" = "cpu" ]; then
        return
    fi
    
    print_status "Checking GPU availability..."
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        print_success "GPU detected: $GPU_INFO"
        
        # Check CUDA version
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9.]*\).*/\1/')
        print_status "CUDA Version: $CUDA_VERSION"
    else
        print_warning "No NVIDIA GPU detected. Switching to CPU mode."
        MODE="cpu"
    fi
}

# Setup Python environment
setup_python_env() {
    print_status "Setting up Python virtual environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel >/dev/null 2>&1
    print_success "Python environment ready"
}

# Install PyTorch
install_pytorch() {
    print_status "Installing PyTorch for $MODE mode..."
    
    # Remove existing PyTorch installations
    pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
    
    if [ "$MODE" = "gpu" ]; then
        # Install GPU PyTorch
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 >/dev/null 2>&1
        print_success "GPU-enabled PyTorch installed"
        
        # Verify GPU detection
        if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
            print_success "PyTorch GPU support verified: $GPU_NAME"
        else
            print_warning "PyTorch installed but GPU not detected. Check CUDA installation."
        fi
    else
        # Install CPU PyTorch
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu >/dev/null 2>&1
        print_success "CPU PyTorch installed"
    fi
}

# Install other dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Core dependencies
    CORE_DEPS="pandas numpy scikit-learn matplotlib seaborn yfinance gym pyarrow"
    pip install $CORE_DEPS >/dev/null 2>&1
    
    # Additional dependencies for monitoring
    MONITOR_DEPS="plotly dash jupyter ipykernel"
    pip install $MONITOR_DEPS >/dev/null 2>&1
    
    print_success "All dependencies installed"
}

# Install project
install_project() {
    print_status "Installing VariBAD project..."
    
    # Install in development mode
    pip install -e . >/dev/null 2>&1
    
    print_success "Project installed"
}

# Create directory structure
create_directories() {
    print_status "Creating directory structure..."
    
    DIRS="data logs checkpoints results plots"
    for dir in $DIRS; do
        mkdir -p "$dir"
    done
    
    # Create subdirectories
    mkdir -p data/{raw,processed}
    mkdir -p config

    print_success "Directory structure created"
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    
    # Test Python imports
    if python -c "import torch, pandas, numpy, sklearn; print('✓ Core imports OK')" 2>/dev/null; then
        print_success "Core imports working"
    else
        print_error "Core import test failed"
        return 1
    fi
    
    # Test VariBAD imports
    if python -c "from varibad import VariBADVAE, VariBADTrainer; print('✓ VariBAD imports OK')" 2>/dev/null; then
        print_success "VariBAD imports working"
    else
        print_error "VariBAD import test failed"
        return 1
    fi
    
    # Test PyTorch device
    DEVICE_TEST=$(python -c "import torch; print('GPU' if torch.cuda.is_available() else 'CPU')" 2>/dev/null)
    print_status "PyTorch device: $DEVICE_TEST"
    
    print_success "Installation test passed"
}

# Create helper scripts (simplified, no tmux)
create_scripts() {
    print_status "Creating helper scripts..."
    
    # Create activation script
    cat > activate_varibad.sh << 'EOF'
#!/bin/bash
# Quick activation script for VariBAD environment
echo "🧠 Activating VariBAD Portfolio Environment..."
source venv/bin/activate

# Show system info
echo "✅ Environment activated"
echo "📊 System Information:"
echo "  Python: $(python --version)"
python -c "import torch; print('  PyTorch: ' + torch.__version__ + ' (' + ('GPU' if torch.cuda.is_available() else 'CPU') + ')')"
echo ""
echo "🚀 Available commands:"
echo "  python varibad/main.py --help                        # Show help"
echo "  python varibad/main.py --mode data_only              # Process data"  
echo "  python varibad/main.py --mode train                  # Start training"
echo "  python monitor_training.py --mode realtime           # Monitor training"
echo "  python monitor_training.py --mode plot               # Generate plots"
EOF
    chmod +x activate_varibad.sh
    
    # Create direct training script (no tmux)
    cat > start_training.sh << EOF
#!/bin/bash
# Fixed VariBAD Training Script with Debug Output
set -e

echo "🏋️ Starting VariBAD Training with Fixed Config System..."
echo "🔍 Debug: Script location: $(pwd)"
echo "🔍 Debug: Script name: $0"

# Activate environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run setup_complete.sh first"
    exit 1
fi

# Define config file path FIRST
CONFIG_FILE="config/training_configs.conf"
echo "🔍 Debug: Config file path: $CONFIG_FILE"

# Check if config file exists BEFORE proceeding
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    echo "Creating default config file..."
    
    # Create config directory
    mkdir -p config
    
    # Create default config file with proper syntax
    cat > "$CONFIG_FILE" << 'CONFIGEOF'
# VariBAD Training Configuration File
# Edit this file to change training parameters without modifying scripts

# =============================================================================
# ACTIVE CONFIGURATION
# Uncomment ONE configuration block below to use it
# =============================================================================

# POLICY_FOCUSED - For when VAE learns but policy doesn't
# Good for: Fixing poor portfolio performance, better reward learning
ACTIVE_CONFIG="POLICY_FOCUSED"

# POLICY_FOCUSED Configuration
POLICY_FOCUSED_NUM_ITERATIONS=300
POLICY_FOCUSED_EPISODE_LENGTH=60
POLICY_FOCUSED_EPISODES_PER_ITERATION=12
POLICY_FOCUSED_VAE_UPDATES=5
POLICY_FOCUSED_LATENT_DIM=3
POLICY_FOCUSED_EXTRA_ARGS="--short_selling"

# QUICK_TEST - Fast test run
QUICK_TEST_NUM_ITERATIONS=20
QUICK_TEST_EPISODE_LENGTH=15
QUICK_TEST_EPISODES_PER_ITERATION=3
QUICK_TEST_VAE_UPDATES=3
QUICK_TEST_LATENT_DIM=2
QUICK_TEST_EXTRA_ARGS=""

# DEFAULT_GPU - Balanced training for GPU
DEFAULT_GPU_NUM_ITERATIONS=500
DEFAULT_GPU_EPISODE_LENGTH=60
DEFAULT_GPU_EPISODES_PER_ITERATION=10
DEFAULT_GPU_VAE_UPDATES=15
DEFAULT_GPU_LATENT_DIM=8
DEFAULT_GPU_EXTRA_ARGS="--short_selling"

# VAE_FOCUSED - More VAE training
VAE_FOCUSED_NUM_ITERATIONS=400
VAE_FOCUSED_EPISODE_LENGTH=45
VAE_FOCUSED_EPISODES_PER_ITERATION=8
VAE_FOCUSED_VAE_UPDATES=25
VAE_FOCUSED_LATENT_DIM=10
VAE_FOCUSED_EXTRA_ARGS="--short_selling"

# CPU_OPTIMIZED - For CPU training
CPU_OPTIMIZED_NUM_ITERATIONS=150
CPU_OPTIMIZED_EPISODE_LENGTH=30
CPU_OPTIMIZED_EPISODES_PER_ITERATION=4
CPU_OPTIMIZED_VAE_UPDATES=6
CPU_OPTIMIZED_LATENT_DIM=4
CPU_OPTIMIZED_EXTRA_ARGS=""

# Advanced settings
DEVICE_PREFERENCE="auto"
EXPERIMENT_NAME="default_experiment"
EXPERIMENT_DESCRIPTION="VariBAD training run"
CONFIGEOF
    
    echo "✅ Created default config file: $CONFIG_FILE"
fi

# Parse config file and load values BEFORE calling python
echo "📖 Loading configuration from: $CONFIG_FILE"

if [ -f "$CONFIG_FILE" ]; then
    # Extract ACTIVE_CONFIG
    ACTIVE_CONFIG=$(grep -E "^ACTIVE_CONFIG=" "$CONFIG_FILE" | head -1 | cut -d'=' -f2 | tr -d '"' | tr -d "'")
    
    if [ -z "$ACTIVE_CONFIG" ]; then
        echo "❌ No ACTIVE_CONFIG found in $CONFIG_FILE. Using POLICY_FOCUSED as default."
        ACTIVE_CONFIG="POLICY_FOCUSED"
    fi
    
    echo "🔧 Using configuration: $ACTIVE_CONFIG"
    
    # Function to extract config values
    get_config_value() {
        local param_name="${ACTIVE_CONFIG}_${1}"
        local default_value="$2"
        
        # Look for the parameter in the config file
        local value=$(grep -E "^${param_name}=" "$CONFIG_FILE" | head -1 | cut -d'=' -f2 | tr -d '"' | tr -d "'")
        
        if [ -z "$value" ]; then
            echo "$default_value"
        else
            echo "$value"
        fi
    }
    
    # Load parameters from active config
    NUM_ITERATIONS=$(get_config_value "NUM_ITERATIONS" "100")
    EPISODE_LENGTH=$(get_config_value "EPISODE_LENGTH" "30")
    EPISODES_PER_ITERATION=$(get_config_value "EPISODES_PER_ITERATION" "5")
    VAE_UPDATES=$(get_config_value "VAE_UPDATES" "10")
    LATENT_DIM=$(get_config_value "LATENT_DIM" "5")
    EXTRA_ARGS=$(get_config_value "EXTRA_ARGS" "")
    
    # Load device preference
    DEVICE_PREFERENCE=$(grep -E "^DEVICE_PREFERENCE=" "$CONFIG_FILE" | head -1 | cut -d'=' -f2 | tr -d '"' | tr -d "'")
    if [ -z "$DEVICE_PREFERENCE" ]; then
        DEVICE_PREFERENCE="auto"
    fi
    
    # Debug output
    echo "🔍 Debug: Loaded config values:"
    echo "   ACTIVE_CONFIG: $ACTIVE_CONFIG"
    echo "   NUM_ITERATIONS: $NUM_ITERATIONS"
    echo "   EPISODE_LENGTH: $EPISODE_LENGTH"
    echo "   EPISODES_PER_ITERATION: $EPISODES_PER_ITERATION"
    echo "   VAE_UPDATES: $VAE_UPDATES"
    echo "   LATENT_DIM: $LATENT_DIM"
    echo "   EXTRA_ARGS: $EXTRA_ARGS"
    echo "   DEVICE_PREFERENCE: $DEVICE_PREFERENCE"
    
else
    echo "❌ Config file not found, using defaults"
    NUM_ITERATIONS=100
    EPISODE_LENGTH=30
    EPISODES_PER_ITERATION=5
    VAE_UPDATES=10
    LATENT_DIM=5
    EXTRA_ARGS=""
    DEVICE_PREFERENCE="auto"
fi

# Determine device
if [ "$DEVICE_PREFERENCE" = "auto" ]; then
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        DEVICE="cuda"
        echo "🔥 GPU detected - using CUDA"
    else
        DEVICE="cpu"
        echo "💻 No GPU detected - using CPU"
    fi
else
    DEVICE="$DEVICE_PREFERENCE"
    echo "🎯 Using specified device: $DEVICE"
fi

# Check if data exists, if not process it first
if [ ! -f "data/sp500_rl_ready_cleaned.parquet" ]; then
    echo "📊 Processing data first..."
    python varibad/main.py --mode data_only
fi

# Build training command with loaded config values
TRAINING_COMMAND="python varibad/main.py --mode train"
TRAINING_COMMAND="$TRAINING_COMMAND --num_iterations $NUM_ITERATIONS"
TRAINING_COMMAND="$TRAINING_COMMAND --episode_length $EPISODE_LENGTH"
TRAINING_COMMAND="$TRAINING_COMMAND --episodes_per_iteration $EPISODES_PER_ITERATION"
TRAINING_COMMAND="$TRAINING_COMMAND --vae_updates $VAE_UPDATES"
TRAINING_COMMAND="$TRAINING_COMMAND --latent_dim $LATENT_DIM"
TRAINING_COMMAND="$TRAINING_COMMAND --device $DEVICE"

# Add extra arguments if present (this is where --short_selling comes from)
if [ ! -z "$EXTRA_ARGS" ] && [ "$EXTRA_ARGS" != '""' ] && [ "$EXTRA_ARGS" != "''" ]; then
    echo "🔍 Debug: Adding extra args: $EXTRA_ARGS"
    TRAINING_COMMAND="$TRAINING_COMMAND $EXTRA_ARGS"
else
    echo "🔍 Debug: No extra args to add (EXTRA_ARGS='$EXTRA_ARGS')"
fi

# Display configuration
echo ""
echo "📋 Training Configuration: $ACTIVE_CONFIG"
echo "  Iterations: $NUM_ITERATIONS"
echo "  Episode Length: $EPISODE_LENGTH"
echo "  Episodes per Iteration: $EPISODES_PER_ITERATION"
echo "  VAE Updates: $VAE_UPDATES"
echo "  Latent Dim: $LATENT_DIM"
echo "  Device: $DEVICE"
echo "  Extra Args: '$EXTRA_ARGS'"
echo ""
echo "🚀 Final Training Command:"
echo "  $TRAINING_COMMAND"
echo ""

# Confirm before proceeding
read -p "🤔 Does this look correct? Press Enter to continue or Ctrl+C to abort..."

# Start training
echo "🎯 Starting training..."
echo "=========================================="

# Execute the training command
eval $TRAINING_COMMAND

echo ""
echo "✅ Training completed!"
echo "📊 Check logs/ directory for training logs"
echo "💾 Check checkpoints/ directory for saved models"
echo "📈 Use ./quick_monitor.sh to analyze results"
EOF
    chmod +x start_training.sh

    # Create monitoring script
    cat > quick_monitor.sh << 'EOF'
#!/bin/bash
# Quick monitoring script
echo "📊 VariBAD Training Monitor"
echo "=========================="

source venv/bin/activate

echo "🔍 Available monitoring options:"
echo "1. Real-time plots:     python monitor_training.py --mode realtime"
echo "2. Generate plots:      python monitor_training.py --mode plot"  
echo "3. Check checkpoints:   python monitor_training.py --mode checkpoints"
echo "4. View logs:           tail -f logs/varibad_pipeline_*.log"

read -p "Choose option (1-4): " choice

case $choice in
    1) python monitor_training.py --mode realtime --interval 60 ;;
    2) python monitor_training.py --mode plot ;;
    3) python monitor_training.py --mode checkpoints ;;
    4) tail -f logs/varibad_pipeline_*.log ;;
    *) echo "Invalid option" ;;
esac
EOF
    chmod +x quick_monitor.sh
    
    # Create data processing script
    cat > process_data.sh << 'EOF'
#!/bin/bash
# Standalone data processing script
echo "📊 Processing S&P 500 Data for VariBAD..."

source venv/bin/activate
python varibad/main.py --mode data_only

echo "✅ Data processing complete!"
echo "📁 Processed data saved to: data/sp500_rl_ready_cleaned.parquet"

# Show data info
if [ -f "data/sp500_rl_ready_cleaned.parquet" ]; then
    python -c "
import pandas as pd
df = pd.read_parquet('data/sp500_rl_ready_cleaned.parquet')
print(f'📊 Dataset Info:')
print(f'   Shape: {df.shape}')
print(f'   Date range: {df[\"date\"].min().date()} to {df[\"date\"].max().date()}')
print(f'   Tickers: {df[\"ticker\"].nunique()}')
print(f'   Features: {len(df.columns)}')
print(f'   Missing values: {df.isnull().sum().sum()}')
"
fi
EOF
    chmod +x process_data.sh
    
    print_success "Helper scripts created"
}

# Create README for quick reference
create_readme() {
    cat > QUICKSTART.md << EOF
# VariBAD Portfolio Optimization - Quick Start

## 🚀 Immediate Next Steps

### 1. Activate Environment
\`\`\`bash
source activate_varibad.sh
\`\`\`

### 2. Process Data (if not done already)
\`\`\`bash
./process_data.sh
\`\`\`

### 3. Start Training
\`\`\`bash
./start_training.sh
\`\`\`

### 4. Monitor Progress (in another terminal)
\`\`\`bash
./quick_monitor.sh
\`\`\`

## 📊 Training Modes

**Current setup: $MODE mode**

- **CPU**: Good for testing, development (~2-3 hours for meaningful results)
- **GPU**: Production training (~30-45 minutes for meaningful results)

## 🔧 Configuration

- **Data**: 30 S&P 500 stocks (1990-2025)
- **Features**: 22+ technical indicators
- **Strategy**: Long/short portfolio optimization
- **Algorithm**: VariBAD (regime-agnostic RL)

## 📈 Expected Results

- **Regime Detection**: Automatic market regime discovery
- **Portfolio Performance**: Risk-adjusted returns via DSR
- **Adaptation**: Dynamic strategy adjustment to market conditions

## 🆘 Need Help?

1. **Check logs**: \`tail -f logs/varibad_pipeline_*.log\`
2. **Monitor training**: \`./quick_monitor.sh\`
3. **Restart**: Stop training (Ctrl+C) and run \`./start_training.sh\` again

---
*Generated by setup_complete.sh on $(date)*
EOF
}

# Main setup function
main() {
    # Step 1: System setup
    update_system
    check_python
    check_gpu
    
    # Step 2: Python environment
    setup_python_env
    install_pytorch
    install_dependencies
    install_project
    
    # Step 3: Project setup
    create_directories
    
    # Step 4: Testing
    if [ "$QUICK_SETUP" = false ]; then
        test_installation
    fi
    
    # Step 5: Helper scripts
    create_scripts
    create_readme
    
    # Final summary
    print_header ""
    print_header "🎉 VARIBAD SETUP COMPLETE!"
    print_header "=========================="
    print_success "Setup completed successfully in $MODE mode"
    print_status "Project ready for training and experimentation"
    
    echo ""
    print_header "📋 NEXT STEPS:"
    echo "1. Activate environment:    source activate_varibad.sh"
    echo "2. Process data:           ./process_data.sh"
    echo "3. Start training:         ./start_training.sh"
    echo "4. Monitor progress:       ./quick_monitor.sh (in another terminal)"
    
    echo ""
    print_header "🔧 TRAINING OPTIONS:"
    if [ "$MODE" = "gpu" ]; then
        echo "• Quick test (5 min):      python varibad/main.py --mode train --num_iterations 10 --device cuda"
        echo "• Development (30 min):    ./start_training.sh"
        echo "• Serious (2 hours):       python varibad/main.py --mode train --num_iterations 2000 --device cuda --short_selling"
    else
        echo "• Quick test (15 min):     python varibad/main.py --mode train --num_iterations 10 --device cpu"
        echo "• Development (2 hours):   ./start_training.sh"
        echo "• Serious (8-12 hours):    python varibad/main.py --mode train --num_iterations 1000 --device cpu"
    fi
    
    echo ""
    print_header "📊 MONITORING:"
    echo "• Real-time plots:         python monitor_training.py --mode realtime"
    echo "• Training logs:           tail -f logs/varibad_pipeline_*.log"
    if [ "$MODE" = "gpu" ]; then
        echo "• GPU usage:               watch -n 1 nvidia-smi"
    fi
    echo "• Generate plots:          python monitor_training.py --mode plot"
    
    echo ""
    print_header "📁 PROJECT STRUCTURE:"
    echo "• Core code:               varibad/"
    echo "• Data files:              data/"
    echo "• Training logs:           logs/"
    echo "• Model checkpoints:       checkpoints/"
    echo "• Results & plots:         results/, plots/"
    echo "• Helper scripts:          *.sh files"
    
    echo ""
    print_header "🚨 IMPORTANT FILES:"
    echo "• activate_varibad.sh      - Activate environment & show info"
    echo "• start_training.sh        - Direct training start (no tmux)"
    echo "• quick_monitor.sh         - Monitor training progress"
    echo "• process_data.sh          - Standalone data processing"
    echo "• QUICKSTART.md            - Quick reference guide"
    
    if [ "$MODE" = "gpu" ]; then
        echo ""
        print_header "⚡ GPU OPTIMIZATIONS ENABLED:"
        echo "• Larger batch sizes for faster training"
        echo "• Extended episode lengths (90 days)"
        echo "• Higher latent dimensions (8-12)"
        echo "• More VAE updates per iteration"
        echo "• Expected training time: 30min - 2 hours"
    else
        echo ""
        print_header "💻 CPU OPTIMIZATIONS ENABLED:"
        echo "• Conservative batch sizes"
        echo "• Moderate episode lengths (30-60 days)"
        echo "• Efficient latent dimensions (5-8)"
        echo "• Balanced VAE updates"
        echo "• Expected training time: 2-12 hours"
    fi
    
    echo ""
    print_success "🎯 Ready to train professional-grade portfolio optimization models!"
    print_status "Start with: source activate_varibad.sh"
}

# Run main setup
main "$@"