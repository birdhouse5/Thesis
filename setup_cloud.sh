#!/bin/bash
# VariBAD Portfolio Cloud Setup Script
# Sets up the complete environment on any Unix-based cloud instance

set -e  # Exit on any error

echo "🚀 VariBAD Portfolio Cloud Setup"
echo "================================="

# Color codes for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

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
    print_status "Updating system packages..."
    OS=$(detect_os)
    
    case $OS in
        "ubuntu")
            sudo apt-get update -y
            sudo apt-get upgrade -y
            sudo apt-get install -y python3 python3-pip python3-venv git wget curl build-essential
            ;;
        "centos")
            sudo yum update -y
            sudo yum install -y python3 python3-pip git wget curl gcc gcc-c++ make
            ;;
        "macos")
            # Assume Homebrew is installed
            brew update
            brew install python3 git wget curl
            ;;
        *)
            print_warning "Unknown OS. Please install Python 3.8+, pip, git, and build tools manually."
            ;;
    esac
    
    print_success "System packages updated"
}

# Set up Python environment
setup_python() {
    print_status "Setting up Python environment..."
    
    # Check Python version
    python3 --version || { print_error "Python 3 not found"; exit 1; }
    
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    print_success "Python environment created"
}

# Install project dependencies
install_dependencies() {
    print_status "Installing project dependencies..."
    
    # Ensure we're in virtual environment
    source venv/bin/activate
    
    # Install PyTorch (CPU version for broad compatibility)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # Install other requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        # Install essential packages if no requirements.txt
        pip install pandas numpy scikit-learn matplotlib seaborn yfinance gym
    fi
    
    # Install the package in development mode
    pip install -e .
    
    print_success "Dependencies installed"
}

# Set up directory structure
setup_directories() {
    print_status "Setting up directory structure..."
    
    mkdir -p {data,logs,checkpoints,results,plots}
    mkdir -p data/{raw,processed}
    
    print_success "Directory structure created"
}

# Configure Git (if not already configured)
setup_git() {
    print_status "Configuring Git..."
    
    # Check if git is already configured
    if ! git config --global user.name >/dev/null 2>&1; then
        echo "Git user not configured. Please provide details:"
        read -p "Enter your Git username: " git_username
        read -p "Enter your Git email: " git_email
        
        git config --global user.name "$git_username"
        git config --global user.email "$git_email"
    fi
    
    # Set up credential caching
    git config --global credential.helper 'cache --timeout=3600'
    
    print_success "Git configured"
}

# Test the installation
test_installation() {
    print_status "Testing installation..."
    
    source venv/bin/activate
    
    # Test Python imports
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" || { print_error "PyTorch import failed"; exit 1; }
    python -c "import pandas; print(f'Pandas: {pandas.__version__}')" || { print_error "Pandas import failed"; exit 1; }
    python -c "import numpy; print(f'NumPy: {numpy.__version__}')" || { print_error "NumPy import failed"; exit 1; }
    
    # Test project imports
    python -c "from varibad import VariBADVAE; print('✅ VariBAD imports work')" || { print_error "VariBAD import failed"; exit 1; }
    
    print_success "Installation test passed"
}

# Create helpful aliases and shortcuts
create_shortcuts() {
    print_status "Creating helpful shortcuts..."
    
    # Create activation script
    cat > activate_varibad.sh << 'EOF'
#!/bin/bash
# Quick activation script for VariBAD environment
echo "🧠 Activating VariBAD Portfolio Environment..."
source venv/bin/activate
echo "✅ Environment activated. Available commands:"
echo "  python varibad/scripts/main.py --help"
echo "  python varibad/scripts/main.py --mode data_only"
echo "  python varibad/scripts/main.py --mode train --num_iterations 100"
EOF
    chmod +x activate_varibad.sh
    
    # Create quick training script
    cat > quick_train.sh << 'EOF'
#!/bin/bash
# Quick training script
echo "🏋️ Starting VariBAD training..."
source venv/bin/activate
python varibad/scripts/main.py --mode train --num_iterations 50 --episode_length 20
EOF
    chmod +x quick_train.sh
    
    print_success "Shortcuts created (activate_varibad.sh, quick_train.sh)"
}

# Main setup function
main() {
    echo "Starting VariBAD Portfolio setup on $(detect_os)..."
    echo "This will take a few minutes..."
    echo ""
    
    # Parse command line arguments
    SKIP_SYSTEM_UPDATE=false
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-system-update)
                SKIP_SYSTEM_UPDATE=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --skip-system-update    Skip system package updates"
                echo "  -h, --help             Show this help message"
                exit 0
                ;;
        esac
    done
    
    # Run setup steps
    if [ "$SKIP_SYSTEM_UPDATE" = false ]; then
        update_system
    else
        print_warning "Skipping system update"
    fi
    
    setup_python
    install_dependencies
    setup_directories
    setup_git
    test_installation
    create_shortcuts
    
    echo ""
    print_success "🎉 VariBAD Portfolio setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Activate environment: source activate_varibad.sh"
    echo "2. Test data processing: python varibad/scripts/main.py --mode data_only"
    echo "3. Start training: python varibad/scripts/main.py --mode train"
    echo "4. Or use quick training: ./quick_train.sh"
    echo ""
    echo "For long training runs, consider using tmux:"
    echo "  tmux new-session -d -s varibad './quick_train.sh'"
    echo "  tmux attach-session -t varibad"
}

# Run main function
main "$@"