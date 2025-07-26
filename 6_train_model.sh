#!/bin/bash
# 6_train_model.sh - Train VariBAD model

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

# Check if data exists
DATA_FILE="data/sp500_rl_ready_cleaned.parquet"
if [ ! -f "$DATA_FILE" ]; then
    print_error "Data file not found: $DATA_FILE"
    print_status "Run './5_process_data.sh' first to create the dataset"
    exit 1
fi

print_status "🏋️ Starting VariBAD Portfolio Optimization Training"

# Parse command line arguments
MODE="quick"
DEVICE="auto"
CONFIG_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "VariBAD Training Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode [quick|dev|full]     Training mode (default: quick)"
            echo "  --device [auto|cpu|gpu]     Device selection (default: auto)"
            echo "  --config CONFIG_NAME        Use specific config (optional)"
            echo "  -h, --help                  Show this help"
            echo ""
            echo "Training Modes:"
            echo "  quick   - 10 iterations, 15 minutes (testing)"
            echo "  dev     - 100 iterations, 1-2 hours (development)"
            echo "  full    - 1000 iterations, 6-12 hours (production)"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Auto-detect device if needed
if [ "$DEVICE" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        DEVICE="cuda"
        print_status "Auto-detected: GPU mode"
    else
        DEVICE="cpu"
        print_status "Auto-detected: CPU mode"
    fi
fi

# Set training parameters based on mode
case $MODE in
    "quick")
        NUM_ITERATIONS=10
        EPISODE_LENGTH=15
        EPISODES_PER_ITER=3
        VAE_UPDATES=3
        LATENT_DIM=3
        BUFFER_SIZE=50
        print_status "Quick mode: ~5-10 minutes training"
        ;;
    "dev")
        NUM_ITERATIONS=100
        EPISODE_LENGTH=30
        EPISODES_PER_ITER=5
        VAE_UPDATES=8
        LATENT_DIM=5
        BUFFER_SIZE=200
        print_status "Development mode: ~1-2 hours training"
        ;;
    "full")
        NUM_ITERATIONS=1000
        EPISODE_LENGTH=60
        EPISODES_PER_ITER=10
        VAE_UPDATES=15
        LATENT_DIM=8
        BUFFER_SIZE=500
        print_status "Full mode: ~6-12 hours training"
        ;;
    *)
        print_error "Invalid mode: $MODE. Use 'quick', 'dev', or 'full'"
        exit 1
        ;;
esac

# Load custom config if specified
if [ -n "$CONFIG_NAME" ] && [ -f "config/training_configs.conf" ]; then
    print_status "Loading custom config: $CONFIG_NAME"
    source config/training_configs.conf
    
    # Override with custom config if it exists
    if grep -q "^${CONFIG_NAME}_" config/training_configs.conf; then
        NUM_ITERATIONS=$(grep "^${CONFIG_NAME}_NUM_ITERATIONS=" config/training_configs.conf | cut -d'=' -f2)
        EPISODE_LENGTH=$(grep "^${CONFIG_NAME}_EPISODE_LENGTH=" config/training_configs.conf | cut -d'=' -f2)
        EPISODES_PER_ITER=$(grep "^${CONFIG_NAME}_EPISODES_PER_ITERATION=" config/training_configs.conf | cut -d'=' -f2)
        VAE_UPDATES=$(grep "^${CONFIG_NAME}_VAE_UPDATES=" config/training_configs.conf | cut -d'=' -f2)
        LATENT_DIM=$(grep "^${CONFIG_NAME}_LATENT_DIM=" config/training_configs.conf | cut -d'=' -f2)
        print_success "Custom config loaded"
    else
        print_warning "Config $CONFIG_NAME not found, using mode defaults"
    fi
fi

# Display training configuration
print_status "Training Configuration:"
echo "  Mode: $MODE"
echo "  Device: $DEVICE"
echo "  Iterations: $NUM_ITERATIONS"
echo "  Episode Length: $EPISODE_LENGTH"
echo "  Episodes/Iteration: $EPISODES_PER_ITER"
echo "  VAE Updates: $VAE_UPDATES"
echo "  Latent Dimension: $LATENT_DIM"
echo "  Buffer Size: $BUFFER_SIZE"

# Check system resources
if [ "$DEVICE" = "cuda" ]; then
    print_status "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
fi

# Create training command
TRAINING_CMD="python varibad/main.py"
TRAINING_CMD="$TRAINING_CMD --mode train"
TRAINING_CMD="$TRAINING_CMD --num_iterations $NUM_ITERATIONS"
TRAINING_CMD="$TRAINING_CMD --episode_length $EPISODE_LENGTH"
TRAINING_CMD="$TRAINING_CMD --episodes_per_iteration $EPISODES_PER_ITER"
TRAINING_CMD="$TRAINING_CMD --vae_updates $VAE_UPDATES"
TRAINING_CMD="$TRAINING_CMD --latent_dim $LATENT_DIM"
TRAINING_CMD="$TRAINING_CMD --buffer_size $BUFFER_SIZE"
TRAINING_CMD="$TRAINING_CMD --device $DEVICE"

# Add short selling for non-quick modes
if [ "$MODE" != "quick" ]; then
    TRAINING_CMD="$TRAINING_CMD --short_selling"
fi

# Display and confirm training command
print_status "Training command:"
echo "  $TRAINING_CMD"
echo ""

# Ask for confirmation for long training runs
if [ "$MODE" = "full" ]; then
    read -p "This will run for 6-12 hours. Continue? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Training cancelled"
        exit 0
    fi
fi

# Start training
print_status "Starting training..."
echo "📊 Monitor progress with: python monitor_training.py --mode realtime"
echo "📋 View logs with: tail -f logs/varibad_pipeline_*.log"
echo "⏹️  Stop training with: Ctrl+C"
echo ""

# Create timestamp for this run
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
echo "Training started at: $(date)" > "logs/training_run_${TIMESTAMP}.info"
echo "Command: $TRAINING_CMD" >> "logs/training_run_${TIMESTAMP}.info"
echo "Mode: $MODE" >> "logs/training_run_${TIMESTAMP}.info"
echo "Device: $DEVICE" >> "logs/training_run_${TIMESTAMP}.info"

# Run training with error handling
if eval $TRAINING_CMD; then
    print_success "Training completed successfully!"
    
    # Show results
    if ls checkpoints/varibad_final_*.pt 1> /dev/null 2>&1; then
        LATEST_CHECKPOINT=$(ls -t checkpoints/varibad_final_*.pt | head -1)
        print_success "Model saved: $LATEST_CHECKPOINT"
    fi
    
    if ls logs/varibad_pipeline_*.log 1> /dev/null 2>&1; then
        LATEST_LOG=$(ls -t logs/varibad_pipeline_*.log | head -1)
        print_status "Training log: $LATEST_LOG"
    fi
    
    print_status "Next steps:"
    echo "  • Evaluate model: python varibad/main.py --mode evaluate --checkpoint [path]"
    echo "  • Generate plots: python monitor_training.py --mode plot"
    echo "  • Archive results: python archive_results.py"
    
else
    print_error "Training failed!"
    print_status "Check the logs for details:"
    if ls logs/varibad_pipeline_*.log 1> /dev/null 2>&1; then
        LATEST_LOG=$(ls -t logs/varibad_pipeline_*.log | head -1)
        echo "  tail -50 $LATEST_LOG"
    fi
    exit 1
fi