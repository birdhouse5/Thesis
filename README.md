# VariBAD Portfolio Optimization

Clean and streamlined implementation of VariBAD for portfolio optimization on S&P 500 data.

## Quick Start

### 1. Setup Environment
```bash
git clone <repository>
cd varibad-portfolio
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Create Configuration
Create your experiment config in `config/my_experiment.conf`:
```json
{
  "experiment_name": "my_test",
  "training": {
    "num_iterations": 100,
    "episode_length": 30,
    "episodes_per_iteration": 5,
    "vae_updates": 10
  },
  "model": {
    "latent_dim": 5,
    "encoder_hidden": 128,
    "decoder_hidden": 128,
    "policy_hidden": 256
  },
  "portfolio": {
    "short_selling": true,
    "max_short_ratio": 0.3,
    "transaction_cost": 0.001
  },
  "environment": {
    "device": "auto"
  }
}
```

### 3. Run Experiment
```bash
python run_experiment.py config/my_experiment.conf
```

### 4. Results
Complete experimental results are automatically saved as:
```
results/my_test_20250127_143022.zip
```

## What's Included in Results

Each experiment zip contains:
- `config.json` - Complete experiment configuration
- `model_checkpoint.pt` - Trained model weights and optimizer states
- `training_stats.json` - All training metrics (losses, rewards, etc.)
- `experiment_metadata.json` - Summary statistics and run info
- `README.md` - Experiment documentation

## Repository Structure

```
varibad-portfolio/
├── config/                    # Experiment configurations
│   ├── experiment1.conf
│   ├── experiment2.conf
│   └── experiment3.conf
├── varibad/
│   ├── __init__.py
│   ├── data.py               # Data loading/preprocessing
│   ├── models.py             # VariBAD VAE, Policy, Environment
│   ├── trainer.py            # Training loop
│   └── utils.py              # Buffer, utilities
├── run_experiment.py         # Main entry point
├── requirements.txt
├── .gitignore
└── README.md
```

## Configuration Options

### Training Parameters
- `num_iterations`: Total training iterations
- `episode_length`: Trading horizon length
- `episodes_per_iteration`: Episodes collected per iteration
- `vae_updates`: VAE training steps per iteration
- `buffer_size`: Maximum episodes in replay buffer
- `eval_frequency`: Evaluation frequency

### Model Parameters
- `latent_dim`: Belief representation dimension
- `encoder_hidden`: Encoder RNN hidden size
- `decoder_hidden`: Decoder network hidden size
- `policy_hidden`: Policy network hidden size

### Portfolio Parameters
- `short_selling`: Enable/disable short positions
- `max_short_ratio`: Maximum short leverage
- `transaction_cost`: Proportional trading cost

### Learning Rates
- `policy_lr`: Policy network learning rate
- `vae_encoder_lr`: VAE encoder learning rate
- `vae_decoder_lr`: VAE decoder learning rate

## Examples

```bash
# Quick test (5-10 minutes)
python run_experiment.py config/experiment1.conf

# Standard training (1-2 hours)
python run_experiment.py config/experiment2.conf

# Long training (4-8 hours)
python run_experiment.py config/experiment3.conf

# Multiple experiments
python run_experiment.py config/experiment1.conf config/experiment2.conf

# Override device
python run_experiment.py config/experiment1.conf --device cuda
```

## Data

The system automatically downloads and preprocesses S&P 500 data for 30 companies (1990-2025) including:
- OHLCV price data
- Technical indicators (RSI, Bollinger Bands, MACD, etc.)
- Market features (volatility, correlations)
- Normalized features for RL training

## Models

**VariBAD VAE**: Variational autoencoder for regime detection
- **Encoder**: RNN that maps trajectory sequences to belief parameters
- **Decoder**: MLP that predicts future states/rewards from beliefs
- **Policy**: MLP that maps state + belief to portfolio weights

**Environment**: Portfolio MDP with realistic constraints
- State: Technical indicators + market features + portfolio weights
- Action: Long/short portfolio weights with leverage constraints
- Reward: Differential Sharpe Ratio (DSR)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- 4GB+ RAM (8GB+ recommended for larger experiments)
- GPU optional but recommended for faster training