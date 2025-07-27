# VariBAD Configuration System

## Overview
This enhanced configuration system enables systematic experimentation and hyperparameter optimization for VariBAD portfolio optimization.

## Structure
```
config/
├── base.conf              # Default values for all parameters
├── profiles/               # Pre-configured profiles for different use cases
│   ├── debug.conf         # Fast iteration (5 iter, 10 episodes)
│   ├── development.conf   # Normal experiments (100 iter, 30 episodes)
│   ├── production.conf    # Long runs (2000 iter, 90 episodes)
│   └── ablation.conf      # Systematic studies
└── experiments/           # Specific research experiments
    ├── exp_001_baseline.conf
    ├── exp_002_no_short.conf
    ├── exp_003_latent_dim_sweep.conf
    └── exp_004_episode_length_study.conf
```

## Usage

### Profile-based Training
```bash
# Quick debugging
python varibad/main.py --config profiles/debug.conf

# Standard development
python varibad/main.py --config profiles/development.conf

# Production training
python varibad/main.py --config profiles/production.conf
```

### Specific Experiments
```bash
# Run baseline experiment
python varibad/main.py --config experiments/exp_001_baseline.conf

# Long-only portfolio study
python varibad/main.py --config experiments/exp_002_no_short.conf
```

### Parameter Sweeps
```bash
# Latent dimension sweep
python varibad/main.py --sweep latent_dim=3,5,8,12 --base profiles/development.conf

# Multiple parameter sweep
python varibad/main.py --sweep episode_length=30,60,90 vae_updates=5,10,15
```

### Automatic Experiment Naming
Experiments are automatically named with timestamp and key parameters:
- exp_20250127_143022_latent5_iter100
- exp_20250127_144523_debug_noShort

## Configuration Format

All configurations use JSON format with hierarchical sections:

### Core Sections
- varibad: VariBAD-specific parameters (latent_dim, hidden sizes)
- training: Training loop parameters (iterations, episodes, updates)
- learning_rates: All learning rate settings
- portfolio: Portfolio constraints and settings
- environment: Environment and device settings
- experiment: Experiment metadata and tracking
- logging: Output and logging configuration

### Parameter Inheritance
Configurations inherit from base.conf and can override any parameter:

1. base.conf provides all defaults
2. Profile configs override relevant sections
3. Experiment configs override specific parameters
4. Command line arguments override everything

### Parameter Sweeps
Use SWEEP:[value1,value2,value3] syntax in configs to define parameter ranges:

```json
{
  "varibad": {
    "latent_dim": "SWEEP:[3,5,8,12]"
  },
  "training": {
    "episode_length": "SWEEP:[30,60,90]"
  }
}
```

## Key Parameters for VariBAD

### VariBAD-Specific
- latent_dim: [3,5,8,12,16] - Belief representation complexity
- kl_regularization_weight: [0.1,1.0,10.0] - Prior strength
- max_trajectory_length: [10,20,50] - Sequence length for encoder

### Training Dynamics
- episode_length: [15,30,60,90] - Trading horizon length
- episodes_per_iteration: [3,5,10] - Data collection rate
- vae_updates: [5,10,20] - VAE learning frequency

### Portfolio Constraints
- short_selling: [True,False] - Enable/disable short positions
- max_short_ratio: [0.1,0.3,0.5] - Maximum leverage
- transaction_cost: [0.0,0.001,0.005] - Trading costs

## Results Tracking

All experiments are automatically tracked in:
- results/experiments/ - Individual experiment results
- results/comparisons/ - Cross-experiment analysis
- SQLite database for systematic querying

## Best Practices

1. Start with profiles: Use debug → development → production progression
2. Systematic sweeps: Use experiment configs for parameter studies
3. Meaningful names: Use descriptive experiment names and tags
4. Document hypotheses: Include research objectives in experiment configs
5. Compare systematically: Use the comparison tools for analysis

## Examples

See config/experiments/ for research-ready experiment configurations.
