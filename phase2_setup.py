#!/usr/bin/env python3
"""
Phase 2: Configuration System Setup - ENCODING FIXED
Creates the enhanced configuration directory structure
"""

import os
from pathlib import Path
import json
from datetime import datetime

def create_config_structure():
    """Create the complete configuration directory structure"""
    
    print("Phase 2: Creating Enhanced Configuration System")
    print("=" * 50)
    
    # Base configuration directories
    config_dirs = [
        "config",
        "config/profiles", 
        "config/experiments",
        "config/templates",
        "results/experiments",
        "results/comparisons"
    ]
    
    for dir_path in config_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    return config_dirs

def create_base_config():
    """Create the base configuration file with all default values"""
    
    base_config = {
        "_meta": {
            "description": "VariBAD Portfolio Optimization - Base Configuration",
            "version": "2.0",
            "created": datetime.now().isoformat(),
            "schema_version": "1.0"
        },
        
        # Core VariBAD Parameters
        "varibad": {
            "latent_dim": 5,
            "encoder_hidden": 128,
            "decoder_hidden": 128, 
            "policy_hidden": 256,
            "kl_regularization_weight": 1.0,
            "max_trajectory_length": 50
        },
        
        # Training Parameters
        "training": {
            "num_iterations": 1000,
            "episode_length": 30,
            "episodes_per_iteration": 5,
            "vae_updates": 10,
            "buffer_size": 500,
            "eval_frequency": 50,
            "save_frequency": 100
        },
        
        # Learning Rates
        "learning_rates": {
            "policy_lr": 1e-4,
            "vae_encoder_lr": 1e-4,
            "vae_decoder_lr": 1e-4,
            "lr_decay_schedule": "constant",
            "lr_decay_factor": 0.95,
            "lr_decay_frequency": 100
        },
        
        # Portfolio Parameters
        "portfolio": {
            "short_selling": True,
            "max_short_ratio": 0.3,
            "transaction_cost": 0.001,
            "initial_capital": 1000000.0,
            "risk_free_rate": 0.02
        },
        
        # Environment Settings
        "environment": {
            "lookback_window": 20,
            "device": "auto",
            "seed": 42,
            "data_path": "data/sp500_rl_ready_cleaned.parquet"
        },
        
        # Experiment Tracking
        "experiment": {
            "name": "default",
            "description": "Default VariBAD configuration",
            "tags": ["baseline"],
            "auto_naming": True,
            "save_frequency": 100
        },
        
        # Logging and Output
        "logging": {
            "log_level": "INFO",
            "console_output": True,
            "file_output": True,
            "tensorboard": False,
            "wandb": False
        }
    }
    
    config_path = Path("config/base.conf")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(base_config, f, indent=2)
    
    print(f"Created base configuration: {config_path}")
    return config_path

def create_profile_configs():
    """Create profile configurations for different use cases"""
    
    profiles = {
        "debug": {
            "_meta": {
                "description": "Debug profile - Fast iteration for development",
                "use_case": "Development and debugging"
            },
            "training": {
                "num_iterations": 5,
                "episode_length": 10,
                "episodes_per_iteration": 2,
                "vae_updates": 2,
                "buffer_size": 50,
                "eval_frequency": 2,
                "save_frequency": 3
            },
            "varibad": {
                "latent_dim": 3,
                "encoder_hidden": 64,
                "decoder_hidden": 64,
                "policy_hidden": 128
            },
            "experiment": {
                "name": "debug",
                "description": "Debug run",
                "tags": ["debug", "development"]
            }
        },
        
        "development": {
            "_meta": {
                "description": "Development profile - Normal experiments", 
                "use_case": "Standard development and testing"
            },
            "training": {
                "num_iterations": 100,
                "episode_length": 30,
                "episodes_per_iteration": 5,
                "vae_updates": 8,
                "buffer_size": 200,
                "eval_frequency": 20,
                "save_frequency": 50
            },
            "varibad": {
                "latent_dim": 5,
                "encoder_hidden": 128,
                "decoder_hidden": 128,
                "policy_hidden": 256
            },
            "experiment": {
                "name": "development",
                "description": "Development experiment",
                "tags": ["development", "standard"]
            }
        },
        
        "production": {
            "_meta": {
                "description": "Production profile - Long training runs",
                "use_case": "Final model training for paper results"
            },
            "training": {
                "num_iterations": 2000,
                "episode_length": 90,
                "episodes_per_iteration": 10,
                "vae_updates": 15,
                "buffer_size": 1000,
                "eval_frequency": 100,
                "save_frequency": 200
            },
            "varibad": {
                "latent_dim": 8,
                "encoder_hidden": 256,
                "decoder_hidden": 256,
                "policy_hidden": 512
            },
            "portfolio": {
                "short_selling": True,
                "max_short_ratio": 0.5
            },
            "experiment": {
                "name": "production",
                "description": "Production training run",
                "tags": ["production", "paper", "final"]
            }
        },
        
        "ablation": {
            "_meta": {
                "description": "Ablation study profile - Systematic component testing",
                "use_case": "Ablation studies and component analysis"
            },
            "training": {
                "num_iterations": 500,
                "episode_length": 60,
                "episodes_per_iteration": 8,
                "vae_updates": 10,
                "buffer_size": 400,
                "eval_frequency": 50,
                "save_frequency": 100
            },
            "experiment": {
                "name": "ablation",
                "description": "Ablation study",
                "tags": ["ablation", "systematic", "analysis"]
            }
        }
    }
    
    profile_paths = []
    for profile_name, config in profiles.items():
        profile_path = Path(f"config/profiles/{profile_name}.conf")
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        profile_paths.append(profile_path)
        print(f"Created profile: {profile_path}")
    
    return profile_paths

def create_experiment_configs():
    """Create specific experiment configurations"""
    
    experiments = {
        "exp_001_baseline": {
            "_meta": {
                "description": "Baseline VariBAD experiment",
                "objective": "Establish baseline performance",
                "hypothesis": "Standard VariBAD should achieve positive Sharpe ratio"
            },
            "experiment": {
                "name": "baseline",
                "description": "Baseline VariBAD with standard parameters",
                "tags": ["baseline", "reference"]
            }
        },
        
        "exp_002_no_short": {
            "_meta": {
                "description": "Long-only portfolio constraint",
                "objective": "Compare long-only vs long-short performance",
                "hypothesis": "Short selling improves risk-adjusted returns"
            },
            "portfolio": {
                "short_selling": False
            },
            "experiment": {
                "name": "no_short",
                "description": "Long-only portfolio experiment",
                "tags": ["long_only", "constraint", "ablation"]
            }
        },
        
        "exp_003_latent_dim_sweep": {
            "_meta": {
                "description": "Latent dimension sweep",
                "objective": "Find optimal belief complexity",
                "hypothesis": "Higher latent dimension improves regime detection"
            },
            "varibad": {
                "latent_dim": "SWEEP:[3,5,8,12,16]"
            },
            "experiment": {
                "name": "latent_sweep",
                "description": "Latent dimension parameter sweep",
                "tags": ["sweep", "latent_dim", "hyperparameter"]
            }
        },
        
        "exp_004_episode_length_study": {
            "_meta": {
                "description": "Trading horizon analysis",
                "objective": "Optimize episode length for portfolio performance",
                "hypothesis": "Longer episodes improve strategy coherence"
            },
            "training": {
                "episode_length": "SWEEP:[15,30,60,90]"
            },
            "experiment": {
                "name": "episode_length_study",
                "description": "Episode length parameter sweep",
                "tags": ["sweep", "episode_length", "horizon"]
            }
        }
    }
    
    experiment_paths = []
    for exp_name, config in experiments.items():
        exp_path = Path(f"config/experiments/{exp_name}.conf")
        with open(exp_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        experiment_paths.append(exp_path)
        print(f"Created experiment: {exp_path}")
    
    return experiment_paths

def create_config_readme():
    """Create documentation for the configuration system - ENCODING FIXED"""
    
    readme_content = """# VariBAD Configuration System

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
"""
    
    readme_path = Path("config/README.md")
    # FIXED: Added encoding parameter
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"Created configuration documentation: {readme_path}")
    return readme_path

def main():
    """Create the complete Phase 2 configuration system"""
    
    # Step 1: Create directory structure
    config_dirs = create_config_structure()
    
    # Step 2: Create base configuration
    base_path = create_base_config()
    
    # Step 3: Create profile configurations
    profile_paths = create_profile_configs()
    
    # Step 4: Create experiment configurations  
    experiment_paths = create_experiment_configs()
    
    # Step 5: Create documentation
    readme_path = create_config_readme()
    
    print("\n" + "=" * 50)
    print("Phase 2 Configuration System Created!")
    print("=" * 50)
    print(f"Created {len(config_dirs)} directories")
    print(f"Created base configuration: {base_path}")
    print(f"Created {len(profile_paths)} profiles")
    print(f"Created {len(experiment_paths)} experiments")
    print(f"Created documentation: {readme_path}")
    
    print("\nReady for enhanced training!")
    print("\nQuick start:")
    print("  python varibad/main.py --config profiles/debug.conf")
    print("  python varibad/main.py --config profiles/development.conf")
    print("  python varibad/main.py --config experiments/exp_001_baseline.conf")
    
    return {
        'base_config': base_path,
        'profiles': profile_paths,
        'experiments': experiment_paths,
        'documentation': readme_path
    }

if __name__ == "__main__":
    result = main()