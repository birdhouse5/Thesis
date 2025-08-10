# generate_configs.py
import json
import itertools
from pathlib import Path
import random

def create_base_config():
    """Base configuration template optimized for 20GB GPU"""
    return {
        "data_path": "environments/data/sp500_rl_ready_cleaned.parquet",
        "train_end": "2015-12-31",
        "val_end": "2020-12-31",
        "num_assets": 30,
        
        # High-capacity defaults for strong GPU
        "batch_size": 1024,
        "vae_batch_size": 512,
        "episodes_per_task": 500,
        "max_episodes": 10000,
        
        # PPO parameters
        "ppo_epochs": 4,
        "ppo_clip_ratio": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "gae_lambda": 0.95,
        "discount_factor": 0.99,
        
        # Context parameters
        "seq_len": 60,
        "min_horizon": 45,
        "max_horizon": 60,
        
        # Validation and testing
        "val_interval": 500,
        "val_episodes": 50,
        "test_episodes": 100,
        "log_interval": 50,
        "save_interval": 1000,
        
        # Device
        "device": "cuda",
        
        # VAE defaults
        "vae_beta": 0.1,
        "vae_update_freq": 1
    }

def generate_architecture_sweep():
    """Comprehensive architecture sweep for 20GB GPU"""
    base = create_base_config()
    configs = []
    
    # Architecture combinations
    architectures = [
        # Small models
        {"latent_dim": 64, "hidden_dim": 256},
        {"latent_dim": 64, "hidden_dim": 512},
        {"latent_dim": 128, "hidden_dim": 256},
        
        # Medium models  
        {"latent_dim": 128, "hidden_dim": 512},
        {"latent_dim": 128, "hidden_dim": 1024},
        {"latent_dim": 256, "hidden_dim": 512},
        
        # Large models
        {"latent_dim": 256, "hidden_dim": 1024},
        {"latent_dim": 256, "hidden_dim": 2048},
        {"latent_dim": 512, "hidden_dim": 1024},
        {"latent_dim": 512, "hidden_dim": 2048},
    ]
    
    # Learning rate combinations
    learning_rates = [
        {"vae_lr": 1e-4, "policy_lr": 1e-4},
        {"vae_lr": 1e-4, "policy_lr": 3e-4},
        {"vae_lr": 3e-4, "policy_lr": 3e-4},
        {"vae_lr": 3e-4, "policy_lr": 1e-3},
        {"vae_lr": 1e-3, "policy_lr": 1e-3},
    ]
    
    exp_id = 1
    for arch in architectures:
        for lr in learning_rates:
            config = base.copy()
            config.update(arch)
            config.update(lr)
            config["exp_name"] = f"arch_{exp_id:03d}_l{arch['latent_dim']}_h{arch['hidden_dim']}_vlr{lr['vae_lr']:.0e}_plr{lr['policy_lr']:.0e}"
            configs.append((f"config_arch_{exp_id:03d}.json", config))
            exp_id += 1
    
    return configs

def generate_vae_comprehensive_sweep():
    """Comprehensive VAE parameter sweep"""
    base = create_base_config()
    base.update({"latent_dim": 256, "hidden_dim": 1024})  # Large model
    
    vae_params = [
        # Beta variations
        {"vae_beta": 0.001, "vae_update_freq": 1, "vae_batch_size": 512},
        {"vae_beta": 0.01, "vae_update_freq": 1, "vae_batch_size": 512},
        {"vae_beta": 0.1, "vae_update_freq": 1, "vae_batch_size": 512},
        {"vae_beta": 0.5, "vae_update_freq": 1, "vae_batch_size": 512},
        {"vae_beta": 1.0, "vae_update_freq": 1, "vae_batch_size": 512},
        
        # Update frequency variations
        {"vae_beta": 0.1, "vae_update_freq": 1, "vae_batch_size": 512},
        {"vae_beta": 0.1, "vae_update_freq": 2, "vae_batch_size": 512},
        {"vae_beta": 0.1, "vae_update_freq": 5, "vae_batch_size": 512},
        {"vae_beta": 0.1, "vae_update_freq": 10, "vae_batch_size": 512},
        
        # Batch size variations (with high capacity)
        {"vae_beta": 0.1, "vae_update_freq": 1, "vae_batch_size": 256},
        {"vae_beta": 0.1, "vae_update_freq": 1, "vae_batch_size": 512},
        {"vae_beta": 0.1, "vae_update_freq": 1, "vae_batch_size": 1024},
        
        # Learning rate variations
        {"vae_beta": 0.1, "vae_update_freq": 1, "vae_batch_size": 512, "vae_lr": 1e-5},
        {"vae_beta": 0.1, "vae_update_freq": 1, "vae_batch_size": 512, "vae_lr": 1e-4},
        {"vae_beta": 0.1, "vae_update_freq": 1, "vae_batch_size": 512, "vae_lr": 1e-3},
        {"vae_beta": 0.1, "vae_update_freq": 1, "vae_batch_size": 512, "vae_lr": 3e-3},
    ]
    
    configs = []
    for i, params in enumerate(vae_params):
        config = base.copy()
        config.update(params)
        
        # Build experiment name
        name_parts = [f"vae_{i+1:03d}"]
        if "vae_lr" in params:
            name_parts.append(f"lr{params['vae_lr']:.0e}")
        name_parts.extend([
            f"beta{params['vae_beta']}",
            f"freq{params['vae_update_freq']}",
            f"batch{params['vae_batch_size']}"
        ])
        
        config["exp_name"] = "_".join(name_parts)
        configs.append((f"config_vae_{i+1:03d}.json", config))
    
    return configs

def generate_context_comprehensive_sweep():
    """Comprehensive context length sweep"""
    base = create_base_config()
    base.update({"latent_dim": 256, "hidden_dim": 1024, "vae_batch_size": 512})
    
    context_params = [
        # Short context
        {"seq_len": 30, "max_horizon": 20, "episodes_per_task": 800},
        {"seq_len": 45, "max_horizon": 30, "episodes_per_task": 600},
        
        # Medium context
        {"seq_len": 60, "max_horizon": 45, "episodes_per_task": 500},
        {"seq_len": 90, "max_horizon": 60, "episodes_per_task": 400},
        
        # Long context
        {"seq_len": 120, "max_horizon": 90, "episodes_per_task": 300},
        {"seq_len": 150, "max_horizon": 120, "episodes_per_task": 200},
        {"seq_len": 180, "max_horizon": 150, "episodes_per_task": 150},
        
        # Very long context (testing limits)
        {"seq_len": 240, "max_horizon": 180, "episodes_per_task": 100},
    ]
    
    configs = []
    for i, params in enumerate(context_params):
        config = base.copy()
        config.update(params)
        config["exp_name"] = f"ctx_{i+1:03d}_seq{params['seq_len']}_hor{params['max_horizon']}_ept{params['episodes_per_task']}"
        configs.append((f"config_ctx_{i+1:03d}.json", config))
    
    return configs

def generate_batch_size_sweep():
    """Comprehensive batch size sweep"""
    base = create_base_config()
    base.update({"latent_dim": 256, "hidden_dim": 1024})
    
    batch_configs = [
        # Different batch size combinations
        {"batch_size": 512, "vae_batch_size": 256, "episodes_per_task": 400},
        {"batch_size": 1024, "vae_batch_size": 512, "episodes_per_task": 500},
        {"batch_size": 2048, "vae_batch_size": 1024, "episodes_per_task": 600},
        {"batch_size": 4096, "vae_batch_size": 2048, "episodes_per_task": 800},
        
        # Very large batches (testing GPU limits)
        {"batch_size": 8192, "vae_batch_size": 4096, "episodes_per_task": 1000},
    ]
    
    configs = []
    for i, params in enumerate(batch_configs):
        config = base.copy()
        config.update(params)
        config["exp_name"] = f"batch_{i+1:03d}_p{params['batch_size']}_v{params['vae_batch_size']}_ept{params['episodes_per_task']}"
        configs.append((f"config_batch_{i+1:03d}.json", config))
    
    return configs

def generate_random_search(n_experiments=50):
    """Generate comprehensive random hyperparameter combinations"""
    base = create_base_config()
    configs = []
    
    param_ranges = {
        "latent_dim": [64, 128, 256, 512],
        "hidden_dim": [256, 512, 1024, 2048],
        "vae_lr": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
        "policy_lr": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
        "vae_beta": [0.001, 0.01, 0.1, 0.3, 0.5, 1.0],
        "batch_size": [512, 1024, 2048, 4096],
        "vae_batch_size": [256, 512, 1024, 2048],
        "seq_len": [30, 60, 90, 120, 150, 180],
        "episodes_per_task": [200, 400, 600, 800, 1000],
        "vae_update_freq": [1, 2, 5, 10],
        "entropy_coef": [0.001, 0.01, 0.05, 0.1],
        "ppo_epochs": [2, 4, 8],
    }
    
    for i in range(n_experiments):
        config = base.copy()
        
        # Sample random values
        for param, values in param_ranges.items():
            config[param] = random.choice(values)
        
        # Ensure vae_batch_size <= batch_size
        config["vae_batch_size"] = min(config["vae_batch_size"], config["batch_size"] // 2)
        
        # Ensure max_horizon <= seq_len
        config["max_horizon"] = min(config["seq_len"] - 10, config["seq_len"] * 0.8)
        
        config["exp_name"] = f"random_{i+1:03d}_l{config['latent_dim']}_h{config['hidden_dim']}"
        configs.append((f"config_random_{i+1:03d}.json", config))
    
    return configs

def save_configs(configs, output_dir="experiments/configs"):
    """Save config files to directory"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for filename, config in configs:
        filepath = output_path / filename
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"Generated {len(configs)} config files in {output_path}")

def main():
    """Generate experiment configurations"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate experiment configurations")
    parser.add_argument("--sweep_type", choices=["architecture", "vae", "context", "batch", "random", "all"], 
                       default="architecture", help="Type of parameter sweep")
    parser.add_argument("--n_random", type=int, default=50, help="Number of random experiments")
    parser.add_argument("--output_dir", type=str, default="experiments/configs", help="Output directory")
    
    args = parser.parse_args()
    
    all_configs = []
    
    if args.sweep_type == "architecture":
        configs = generate_architecture_sweep()
    elif args.sweep_type == "vae":
        configs = generate_vae_comprehensive_sweep()
    elif args.sweep_type == "context":
        configs = generate_context_comprehensive_sweep()
    elif args.sweep_type == "batch":
        configs = generate_batch_size_sweep()
    elif args.sweep_type == "random":
        configs = generate_random_search(args.n_random)
    elif args.sweep_type == "all":
        # Generate all sweeps
        all_configs.extend(generate_architecture_sweep())
        all_configs.extend(generate_vae_comprehensive_sweep())
        all_configs.extend(generate_context_comprehensive_sweep())
        all_configs.extend(generate_batch_size_sweep())
        all_configs.extend(generate_random_search(args.n_random))
        configs = all_configs
    
    save_configs(configs, args.output_dir)
    
    # Print summary
    print(f"\nGenerated {len(configs)} experiments:")
    
    # Count by type
    type_counts = {}
    for filename, config in configs:
        exp_type = filename.split('_')[1]
        type_counts[exp_type] = type_counts.get(exp_type, 0) + 1
    
    for exp_type, count in type_counts.items():
        print(f"  {exp_type}: {count} experiments")
    
    print(f"\nFirst 5 experiments:")
    for filename, config in configs[:5]:
        print(f"  {filename}: {config['exp_name']}")
    if len(configs) > 5:
        print(f"  ... and {len(configs) - 5} more")
    
    # Estimate total time (rough)
    avg_time_per_exp = 30  # minutes (rough estimate for 10k episodes)
    total_hours = (len(configs) * avg_time_per_exp) / 60
    print(f"\nEstimated total time: ~{total_hours:.1f} hours ({total_hours/24:.1f} days)")


if __name__ == "__main__":
    main()