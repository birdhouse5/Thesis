# VariBAD Portfolio Optimization - Complete Setup Guide

This guide sets up the complete VariBAD portfolio optimization system from scratch. The system implements regime-agnostic reinforcement learning for portfolio optimization using variBAD (Variational Bayes for Adaptive Deep RL).

## Quick Start (One Command Setup)

```bash
# Clone and setup everything automatically
git clone https://github.com/Birdhouse5/Thesis.git
cd Thesis

# For CPU training
./setup_complete.sh --mode cpu

# For GPU training (requires NVIDIA GPU)
./setup_complete.sh --mode gpu

# Start training immediately
./start_training.sh
```

## What This System Does

- **Portfolio Optimization**: Manages a portfolio of 30 S&P 500 stocks (1990-2025)
- **Regime Detection**: Automatically discovers market regimes without manual labeling
- **Technical Indicators**: Uses 22+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Long/Short Trading**: Supports both long-only and long/short strategies
- **Risk Management**: Uses Differential Sharpe Ratio (DSR) for risk-adjusted returns

## Project Structure
```
├── varibad/                    # Core VariBAD implementation
│   ├── core/                   # Neural network models
│   ├── utils/                  # Utilities and buffers
│   ├── scripts/               # Training scripts
│   └── data_pipeline.py       # Data processing
├── data/                      # Processed datasets (auto-generated)
├── logs/                      # Training logs
├── checkpoints/              # Model checkpoints
├── results/                  # Training results
├── plots/                    # Training visualizations
└── setup_complete.sh         # One-command setup script
```

## System Requirements

### Minimum Requirements
- **OS**: Linux/macOS/Windows WSL
- **Python**: 3.8+
- **RAM**: 8GB+ 
- **Storage**: 5GB+ free space

### Recommended (GPU Training)
- **GPU**: NVIDIA GPU with 6GB+ VRAM (RTX 3080/4080, Tesla V100, etc.)
- **CUDA**: 11.8+ or 12.x
- **RAM**: 16GB+

## Detailed Setup Instructions

### 1. Environment Setup

```bash
# Update system packages (Ubuntu/Debian)
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3 python3-pip python3-venv git wget curl build-essential

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Verify Python version
python --version  # Should be 3.8+
```

### 2. Install Dependencies

#### For CPU Training:
```bash
# Install CPU PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install pandas numpy scikit-learn matplotlib seaborn yfinance gym pyarrow
```

#### For GPU Training:
```bash
# Check GPU availability
nvidia-smi

# Install GPU PyTorch (CUDA 12.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install pandas numpy scikit-learn matplotlib seaborn yfinance gym pyarrow

# Verify GPU detection
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### 3. Project Installation

```bash
# Install the project
pip install -e .

# Create directory structure
mkdir -p {data,logs,checkpoints,results,plots}

# Test core imports
python -c "from varibad import VariBADVAE, VariBADTrainer; print('✅ VariBAD imports working')"
```

### 4. Data Processing

The system automatically downloads and processes S&P 500 data:

```bash
# Process data only
python varibad/scripts/main.py --mode data_only

# This will:
# 1. Download 30 S&P 500 stocks (1990-2025) using yfinance
# 2. Calculate 22 technical indicators
# 3. Normalize features for ML training
# 4. Save to data/sp500_rl_ready_cleaned.parquet
```

### 5. Training

#### Quick Test Training (5 minutes):
```bash
python varibad/scripts/main.py --mode train --num_iterations 10 --episode_length 10
```

#### Serious Training:

**CPU Training (8-12 hours):**
```bash
python varibad/scripts/main.py --mode train \
    --num_iterations 1000 \
    --episode_length 60 \
    --episodes_per_iteration 5 \
    --vae_updates 10 \
    --latent_dim 5 \
    --device cpu
```

**GPU Training (1-2 hours):**
```bash
python varibad/scripts/main.py --mode train \
    --num_iterations 2000 \
    --episode_length 90 \
    --episodes_per_iteration 12 \
    --vae_updates 20 \
    --latent_dim 12 \
    --device cuda \
    --short_selling
```

### 6. Monitoring Training

#### Real-time Monitoring:
```bash
# Install monitoring dependencies
pip install matplotlib seaborn

# Start real-time monitoring with plots
python monitor_training.py --mode realtime --interval 60
```

#### Manual Monitoring:
```bash
# View logs
tail -f logs/varibad_pipeline_*.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check training session (if using tmux)
tmux attach-session -t varibad_training
```

### 7. Long Training Sessions

For extended training, use tmux to prevent interruption:

```bash
# Start tmux session
tmux new-session -d -s varibad_training

# Send training command to tmux
tmux send-keys -t varibad_training "source venv/bin/activate" Enter
tmux send-keys -t varibad_training "python varibad/scripts/main.py --mode train --num_iterations 2000 --device cuda" Enter

# Attach to monitor
tmux attach-session -t varibad_training

# Detach: Ctrl+B, then D
# Kill session: tmux kill-session -t varibad_training
```

## Training Parameters Guide

### Quick Test (5 minutes)
```bash
--num_iterations 10
--episode_length 10
--episodes_per_iteration 2
--vae_updates 5
```

### Development Training (30 minutes)
```bash
--num_iterations 100
--episode_length 30
--episodes_per_iteration 5
--vae_updates 10
--latent_dim 5
```

### Serious Training (1-2 hours GPU, 8-12 hours CPU)
```bash
--num_iterations 2000
--episode_length 90
--episodes_per_iteration 12
--vae_updates 20
--latent_dim 12
--short_selling
```

### Production Training (4-6 hours GPU)
```bash
--num_iterations 5000
--episode_length 120
--episodes_per_iteration 15
--vae_updates 25
--latent_dim 16
--buffer_size 2000
--short_selling
```

## Key Parameters Explained

- **num_iterations**: Total training iterations (more = better performance)
- **episode_length**: Trading days per episode (30-120 recommended)
- **episodes_per_iteration**: Parallel episodes collected (higher = more diverse data)
- **vae_updates**: VAE training steps per iteration (important for regime learning)
- **latent_dim**: Dimension of regime representation (5-16 recommended)
- **short_selling**: Enable long/short strategies (more realistic)
- **device**: 'cpu', 'cuda', or 'auto'

## Expected Results

### Performance Metrics
- **DSR (Differential Sharpe Ratio)**: Primary reward metric
- **Portfolio Returns**: Cumulative and per-episode returns
- **Volatility**: Risk measures
- **Regime Detection**: Learned market regimes

### Training Time
- **CPU**: ~6 iterations/minute
- **GPU (RTX 3080)**: ~25-30 iterations/minute
- **GPU (V100)**: ~40-50 iterations/minute

### Memory Usage
- **CPU**: ~2-4GB RAM
- **GPU**: ~4-8GB VRAM (depends on batch size)

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Fix: Reinstall in development mode
pip install -e .
```

#### 2. CUDA Not Available
```bash
# Check GPU
nvidia-smi

# Reinstall GPU PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 3. Out of Memory (GPU)
```bash
# Reduce batch size
--episodes_per_iteration 8
--vae_updates 15
--latent_dim 8
```

#### 4. Slow Training (CPU)
```bash
# Use smaller parameters
--episode_length 30
--episodes_per_iteration 3
--vae_updates 8
```

#### 5. Data Download Issues
```bash
# Clear data and retry
rm -rf data/
python varibad/scripts/main.py --mode data_only
```

### Getting Help

#### Check System Status
```bash
# Verify installation
python -c "from varibad import VariBADVAE; print('✅ Working')"

# Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# View recent logs
tail -20 logs/varibad_pipeline_*.log
```

#### Debug Mode
```bash
# Run with debug logging
python varibad/scripts/main.py --mode train --log_level DEBUG --num_iterations 5
```

## Advanced Usage

### Resume Training
```bash
# Resume from checkpoint
python varibad/scripts/main.py --mode resume --checkpoint checkpoints/varibad_final_20250723_144910.pt
```

### Evaluate Model
```bash
# Evaluate trained model
python varibad/scripts/main.py --mode evaluate --checkpoint checkpoints/varibad_final_20250723_144910.pt
```

### Custom Data
```bash
# Use your own data (must match expected format)
python varibad/scripts/main.py --mode train --data_path your_data.parquet
```

## File Structure After Setup

```
your-repo/
├── data/
│   ├── sp500_constituents.csv              # Stock list
│   ├── sp500_ohlcv_dataset.parquet        # Raw OHLCV data
│   ├── sp500_with_indicators.parquet      # With technical indicators
│   ├── sp500_rl_ready.parquet             # Normalized features
│   └── sp500_rl_ready_cleaned.parquet     # Final clean dataset
├── logs/
│   └── varibad_pipeline_YYYYMMDD_HHMMSS.log
├── checkpoints/
│   └── varibad_final_YYYYMMDD_HHMMSS.pt
├── plots/
│   └── training_progress_YYYYMMDD_HHMMSS.png
└── results/
    └── evaluation_results.json
```

## Next Steps

1. **Start with quick test**: `python varibad/scripts/main.py --mode train --num_iterations 10`
2. **Monitor training**: `python monitor_training.py --mode realtime`
3. **Scale up gradually**: Increase iterations as you verify everything works
4. **Experiment with parameters**: Try different episode lengths, latent dimensions
5. **Analyze results**: Use the monitoring tools to understand regime learning

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review logs in `logs/varibad_pipeline_*.log`
3. Verify system requirements are met
4. Test with minimal parameters first

---

**Ready to start?** Run `./setup_complete.sh --mode gpu` and begin training professional-grade portfolio optimization models!