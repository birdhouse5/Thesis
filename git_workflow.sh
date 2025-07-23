#!/bin/bash
# VariBAD Portfolio - Git Workflow & Cloud Deployment Script
# Handles development workflow and cloud instance deployment

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

show_help() {
    echo "VariBAD Portfolio - Git Workflow Script"
    echo "======================================"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  init           Initialize repository with proper structure"
    echo "  save <msg>     Add, commit, and push changes with message"
    echo "  sync           Pull latest changes from remote"
    echo "  status         Show git status and branch info"
    echo "  deploy         Prepare for cloud deployment"
    echo "  setup-cloud    Set up on a fresh cloud instance"
    echo ""
    echo "Examples:"
    echo "  $0 init"
    echo "  $0 save \"Added new VariBAD features\""
    echo "  $0 deploy"
    echo "  $0 setup-cloud"
}

# Initialize repository with proper structure
init_repo() {
    print_status "Initializing VariBAD Portfolio repository..."
    
    # Initialize git if not already done
    if [ ! -d ".git" ]; then
        git init
        print_success "Git repository initialized"
    fi
    
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
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# Data files (keep structure, ignore large files)
data/*.parquet
data/*.csv
data/raw/
data/processed/

# Training outputs
logs/
checkpoints/*.pt
checkpoints/*.pth
results/
plots/*.png
plots/*.pdf

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints/

# Environment variables
.env
.env.local

# Temporary files
*.tmp
*.temp
*.log
EOF
        print_success ".gitignore created"
    fi
    
    # Create README if it doesn't exist
    if [ ! -f "README.md" ]; then
        cat > README.md << 'EOF'
# VariBAD Portfolio Optimization

Regime-agnostic reinforcement learning for portfolio optimization using variBAD (Variational Bayes for Adaptive Deep RL).

## Quick Start

### Local Development
```bash
# Clone repository
git clone <your-repo-url>
cd varibad-portfolio

# Set up environment
chmod +x setup_cloud.sh
./setup_cloud.sh

# Activate environment
source activate_varibad.sh

# Process data and train
python varibad/scripts/main.py --mode train --num_iterations 100
```

### Cloud Deployment
```bash
# On fresh cloud instance
git clone <your-repo-url>
cd varibad-portfolio
./setup_cloud.sh

# Start training
./quick_train.sh
```

## Project Structure
```
varibad/
├── core/           # Main VariBAD components
├── data/           # Data processing & preprocessing
├── utils/          # Utilities and tools
└── scripts/        # Entry points
```

## Features
- 30 S&P 500 companies, 1990-2025
- 22 technical indicators with robust implementations
- MetaTrader-style portfolio MDP with DSR rewards
- Implicit regime detection through variBAD
- Long/short portfolio optimization

## Training
- **Data only**: `python varibad/scripts/main.py --mode data_only`
- **Full training**: `python varibad/scripts/main.py --mode train --num_iterations 1000`
- **Resume training**: `python varibad/scripts/main.py --mode resume --checkpoint path/to/checkpoint.pt`

## Requirements
- Python 3.8+
- PyTorch 2.0+
- See `requirements.txt` for full dependencies
EOF
        print_success "README.md created"
    fi
    
    print_success "Repository initialization complete"
}

# Save changes with commit message
save_changes() {
    if [ -z "$1" ]; then
        print_error "Commit message required"
        echo "Usage: $0 save \"Your commit message\""
        exit 1
    fi
    
    commit_msg="$1"
    
    print_status "Saving changes to repository..."
    
    # Show what will be committed
    echo ""
    print_status "Files to be committed:"
    git status --porcelain
    echo ""
    
    # Add all changes
    git add .
    
    # Commit with message
    git commit -m "$commit_msg"
    print_success "Changes committed: $commit_msg"
    
    # Push to remote (if configured)
    if git remote get-url origin >/dev/null 2>&1; then
        print_status "Pushing to remote repository..."
        git push origin $(git branch --show-current)
        print_success "Changes pushed to remote"
    else
        print_warning "No remote repository configured. Add one with:"
        echo "  git remote add origin <your-repo-url>"
        echo "  git push -u origin main"
    fi
}

# Sync with remote repository
sync_repo() {
    print_status "Syncing with remote repository..."
    
    if ! git remote get-url origin >/dev/null 2>&1; then
        print_error "No remote repository configured"
        exit 1
    fi
    
    # Fetch latest changes
    git fetch origin
    
    # Check if we're behind
    current_branch=$(git branch --show-current)
    behind=$(git rev-list --count HEAD..origin/$current_branch 2>/dev/null || echo "0")
    
    if [ "$behind" -gt 0 ]; then
        print_status "Pulling $behind commits from remote..."
        git pull origin $current_branch
        print_success "Repository synced"
    else
        print_success "Repository is up to date"
    fi
}

# Show repository status
show_status() {
    print_status "Repository Status"
    echo "=================="
    echo ""
    
    # Branch info
    echo "Branch: $(git branch --show-current)"
    echo "Latest commit: $(git log -1 --pretty=format:'%h - %s (%cr)')"
    echo ""
    
    # Remote info
    if git remote get-url origin >/dev/null 2>&1; then
        echo "Remote: $(git remote get-url origin)"
    else
        echo "Remote: Not configured"
    fi
    echo ""
    
    # Working directory status
    if [ -n "$(git status --porcelain)" ]; then
        echo "Uncommitted changes:"
        git status --short
    else
        echo "Working directory clean"
    fi
    echo ""
    
    # Recent commits
    echo "Recent commits:"
    git log --oneline -5
}

# Prepare for cloud deployment
prepare_deployment() {
    print_status "Preparing for cloud deployment..."
    
    # Ensure all necessary files are present
    required_files=("varibad/scripts/main.py" "requirements.txt" "setup_cloud.sh")
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            print_error "Required file missing: $file"
            exit 1
        fi
    done
    
    # Make scripts executable
    chmod +x setup_cloud.sh
    chmod +x git_workflow.sh
    
    # Create deployment checklist
    cat > DEPLOYMENT.md << 'EOF'
# Cloud Deployment Checklist

## Pre-deployment
- [ ] All code committed and pushed to repository
- [ ] Requirements.txt updated
- [ ] Setup scripts are executable
- [ ] .gitignore excludes large data files

## On Cloud Instance
```bash
# 1. Clone repository
git clone <your-repo-url>
cd varibad-portfolio

# 2. Run setup
./setup_cloud.sh

# 3. Activate environment
source activate_varibad.sh

# 4. Test installation
python -c "from varibad import VariBADVAE; print('✅ Ready!')"

# 5. Start training
./quick_train.sh
```

## For Long Training Sessions
```bash
# Use tmux for persistent sessions
tmux new-session -d -s varibad
tmux send-keys -t varibad './quick_train.sh' Enter
tmux attach-session -t varibad  # To monitor
# Ctrl+B, D to detach
```

## Monitoring Training
```bash
# Check logs
tail -f logs/varibad_pipeline_*.log

# Check GPU usage (if available)
nvidia-smi

# Check training progress
ls -la checkpoints/
```
EOF
    
    print_success "Deployment preparation complete"
    print_status "See DEPLOYMENT.md for cloud setup instructions"
}

# Set up on cloud instance
setup_cloud() {
    print_status "Setting up VariBAD on cloud instance..."
    
    if [ ! -f "setup_cloud.sh" ]; then
        print_error "setup_cloud.sh not found. Run 'deploy' command first."
        exit 1
    fi
    
    # Make executable and run
    chmod +x setup_cloud.sh
    ./setup_cloud.sh
    
    print_success "Cloud setup complete"
}

# Main function
main() {
    case "${1:-help}" in
        "init")
            init_repo
            ;;
        "save")
            save_changes "$2"
            ;;
        "sync")
            sync_repo
            ;;
        "status")
            show_status
            ;;
        "deploy")
            prepare_deployment
            ;;
        "setup-cloud")
            setup_cloud
            ;;
        "help"|"--help"|"-h"|"")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"