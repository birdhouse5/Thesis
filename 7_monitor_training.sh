#!/bin/bash
# 7_monitor_training.sh - Monitor VariBAD training progress

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

print_status "📊 VariBAD Training Monitor"

# Parse command line arguments
MODE="menu"
INTERVAL=60

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        -h|--help)
            echo "VariBAD Training Monitor"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode [menu|realtime|plot|logs|status]  Monitor mode"
            echo "  --interval SECONDS                       Update interval for realtime (default: 60)"
            echo "  -h, --help                               Show this help"
            echo ""
            echo "Modes:"
            echo "  menu     - Interactive menu (default)"
            echo "  realtime - Live training plots and stats"
            echo "  plot     - Generate training plots once"
            echo "  logs     - Show recent log entries"
            echo "  status   - Quick training status check"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to check training status
check_training_status() {
    print_status "Training Status Check"
    
    # Check for running processes
    if pgrep -f "varibad.*train" >/dev/null; then
        print_success "✅ Training process is running"
        echo "  PID: $(pgrep -f 'varibad.*train')"
    else
        print_warning "⚠️  No training process detected"
    fi
    
    # Check for recent logs
    if ls logs/varibad_pipeline_*.log 1> /dev/null 2>&1; then
        LATEST_LOG=$(ls -t logs/varibad_pipeline_*.log | head -1)
        LOG_AGE=$(( $(date +%s) - $(stat -c %Y "$LATEST_LOG" 2>/dev/null || stat -f %m "$LATEST_LOG" 2>/dev/null || echo 0) ))
        
        print_status "Latest log: $LATEST_LOG"
        if [ $LOG_AGE -lt 300 ]; then
            print_success "  📝 Recent activity (${LOG_AGE}s ago)"
        else
            print_warning "  📝 Last activity: ${LOG_AGE}s ago"
        fi
    else
        print_warning "📝 No training logs found"
    fi
    
    # Check for checkpoints
    if ls checkpoints/*.pt 1> /dev/null 2>&1; then
        CHECKPOINT_COUNT=$(ls checkpoints/*.pt | wc -l)
        LATEST_CHECKPOINT=$(ls -t checkpoints/*.pt | head -1)
        print_success "💾 Checkpoints: $CHECKPOINT_COUNT saved"
        print_status "  Latest: $(basename $LATEST_CHECKPOINT)"
    else
        print_warning "💾 No checkpoints found yet"
    fi
    
    # Check GPU usage if available
    if command -v nvidia-smi >/dev/null 2>&1; then
        print_status "🔥 GPU Status:"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        while IFS=, read gpu_util mem_used mem_total; do
            echo "  GPU Utilization: ${gpu_util}%"
            echo "  Memory: ${mem_used}MB / ${mem_total}MB"
        done
    fi
}

# Function to show recent logs
show_recent_logs() {
    print_status "Recent Training Logs"
    
    if ls logs/varibad_pipeline_*.log 1> /dev/null 2>&1; then
        LATEST_LOG=$(ls -t logs/varibad_pipeline_*.log | head -1)
        print_status "Showing last 20 lines from: $LATEST_LOG"
        echo ""
        tail -20 "$LATEST_LOG"
    else
        print_warning "No training logs found"
    fi
}

# Function to generate plots
generate_plots() {
    print_status "Generating training plots..."
    
    if python monitor_training.py --mode plot 2>/dev/null; then
        print_success "Plots generated successfully!"
        
        if ls plots/varibad_dashboard_*.png 1> /dev/null 2>&1; then
            LATEST_PLOT=$(ls -t plots/varibad_dashboard_*.png | head -1)
            print_status "Latest plot: $LATEST_PLOT"
        fi
    else
        print_error "Plot generation failed. Check if monitor_training.py exists and training data is available."
    fi
}

# Function for realtime monitoring
start_realtime_monitoring() {
    print_status "Starting realtime monitoring (update every ${INTERVAL}s)"
    print_status "Press Ctrl+C to stop"
    echo ""
    
    if python monitor_training.py --mode realtime --interval $INTERVAL 2>/dev/null; then
        print_success "Realtime monitoring completed"
    else
        print_error "Realtime monitoring failed. Check if monitor_training.py exists."
        print_status "Falling back to log monitoring..."
        
        # Fallback to simple log monitoring
        while true; do
            clear
            echo "📊 VariBAD Training Monitor - $(date)"
            echo "=" * 50
            check_training_status
            echo ""
            show_recent_logs
            echo ""
            echo "Next update in ${INTERVAL}s... (Ctrl+C to stop)"
            sleep $INTERVAL
        done
    fi
}

# Function for interactive menu
show_menu() {
    while true; do
        clear
        echo "📊 VariBAD Training Monitor"
        echo "=========================="
        echo ""
        echo "1. Check training status"
        echo "2. Show recent logs"
        echo "3. Generate training plots"
        echo "4. Start realtime monitoring"
        echo "5. Follow logs (live)"
        echo "6. Show GPU status"
        echo "7. Exit"
        echo ""
        read -p "Select option [1-7]: " choice
        
        case $choice in
            1)
                clear
                check_training_status
                echo ""
                read -p "Press Enter to continue..."
                ;;
            2)
                clear
                show_recent_logs
                echo ""
                read -p "Press Enter to continue..."
                ;;
            3)
                clear
                generate_plots
                echo ""
                read -p "Press Enter to continue..."
                ;;
            4)
                clear
                start_realtime_monitoring
                break
                ;;
            5)
                clear
                if ls logs/varibad_pipeline_*.log 1> /dev/null 2>&1; then
                    LATEST_LOG=$(ls -t logs/varibad_pipeline_*.log | head -1)
                    print_status "Following log: $LATEST_LOG"
                    print_status "Press Ctrl+C to return to menu"
                    tail -f "$LATEST_LOG"
                else
                    print_warning "No training logs found"
                    read -p "Press Enter to continue..."
                fi
                ;;
            6)
                clear
                if command -v nvidia-smi >/dev/null 2>&1; then
                    nvidia-smi
                else
                    print_warning "nvidia-smi not available"
                fi
                echo ""
                read -p "Press Enter to continue..."
                ;;
            7)
                print_status "Exiting monitor"
                exit 0
                ;;
            *)
                print_error "Invalid option: $choice"
                sleep 1
                ;;
        esac
    done
}

# Execute based on mode
case $MODE in
    "menu")
        show_menu
        ;;
    "realtime")
        start_realtime_monitoring
        ;;
    "plot")
        generate_plots
        ;;
    "logs")
        show_recent_logs
        ;;
    "status")
        check_training_status
        ;;
    *)
        print_error "Invalid mode: $MODE"
        print_status "Available modes: menu, realtime, plot, logs, status"
        exit 1
        ;;
esac