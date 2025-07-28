#!/usr/bin/env python3
"""
Simple experiment runner for VariBAD portfolio optimization
Creates complete experimental archives with all results

ENHANCED: Now supports parameter sweeps with SWEEP: syntax
"""

import argparse
import json
import torch
import shutil
from pathlib import Path
from datetime import datetime
import zipfile
import sys
import os
import itertools

def parse_sweep_config(config):
    """Parse SWEEP: syntax and return list of configs"""
    sweep_params = {}
    
    def find_sweeps(obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, str) and value.startswith("SWEEP:"):
                    sweep_str = value[6:]  # Remove "SWEEP:"
                    try:
                        sweep_values = json.loads(sweep_str)
                        sweep_params[current_path] = sweep_values
                    except:
                        print(f"Invalid sweep format: {value}")
                elif isinstance(value, dict):
                    find_sweeps(value, current_path)
    
    find_sweeps(config)
    
    if not sweep_params:
        return [config]
    
    # Generate all combinations
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    
    configs = []
    for combination in itertools.product(*param_values):
        new_config = json.loads(json.dumps(config))  # Deep copy
        
        for param_name, value in zip(param_names, combination):
            keys = param_name.split('.')
            current = new_config
            for key in keys[:-1]:
                current = current[key]
            current[keys[-1]] = value
        
        configs.append(new_config)
    
    return configs

def run_single_experiment(config, config_name):
    """Run single experiment and save complete results as zip"""
    
    exp_name = config.get('experiment_name', 'experiment')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create results/ directory if it doesn't exist
    results_base = Path("results")
    results_base.mkdir(exist_ok=True)
    
    # Create experiment-specific directory
    exp_dir = results_base / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Results directory: {exp_dir}")
    
    try:
        # Import and initialize trainer
        from varibad.trainer import VariBADTrainer
        
        print("🏗️  Initializing trainer...")
        trainer = VariBADTrainer(config)
        
        print(f"✓ Trainer ready - {trainer.get_model_info()}")
        
        # Run training
        print("🏋️  Starting training...")
        stats = trainer.train()
        
        print("💾 Saving results...")
        
        # Save model checkpoint
        checkpoint_path = exp_dir / "model_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))
        
        # Save configuration
        with open(exp_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save training statistics
        with open(exp_dir / "training_stats.json", 'w') as f:
            json_stats = {}
            for key, value in stats.items():
                if hasattr(value, 'tolist'):
                    json_stats[key] = value.tolist()
                elif isinstance(value, (list, tuple)):
                    json_stats[key] = list(value)
                else:
                    json_stats[key] = value
            json.dump(json_stats, f, indent=2)
        
        # Save experiment metadata
        metadata = {
            'experiment_name': exp_name,
            'timestamp': timestamp,
            'config_name': config_name,
            'total_iterations': len(stats.get('iteration', [])),
            'model_parameters': trainer.get_parameter_count(),
            'device_used': str(trainer.device),
            'final_performance': {
                'avg_episode_reward': stats.get('avg_episode_reward', [])[-1] if stats.get('avg_episode_reward') else None,
                'avg_vae_loss': stats.get('avg_vae_loss', [])[-1] if stats.get('avg_vae_loss') else None
            }
        }
        
        with open(exp_dir / "experiment_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create README
        total_iterations = metadata.get('total_iterations', 'N/A')
        model_parameters = metadata.get('model_parameters', 'N/A')
        device_used = metadata.get('device_used', 'N/A')
        
        final_reward = metadata['final_performance']['avg_episode_reward']
        final_vae_loss = metadata['final_performance']['avg_vae_loss']
        
        final_reward_str = f"{final_reward:.4f}" if final_reward is not None else 'N/A'
        final_vae_loss_str = f"{final_vae_loss:.4f}" if final_vae_loss is not None else 'N/A'
        model_params_str = f"{model_parameters:,}" if isinstance(model_parameters, int) else str(model_parameters)
        
        readme_content = f"""# VariBAD Experiment: {exp_name}

**Timestamp:** {timestamp}
**Config:** {config_name}

## Results:
- Total iterations: {total_iterations}
- Model parameters: {model_params_str}
- Device used: {device_used}
- Final avg episode reward: {final_reward_str}
- Final VAE loss: {final_vae_loss_str}

## Files:
- `config.json` - Experiment configuration
- `model_checkpoint.pt` - Trained model checkpoint
- `training_stats.json` - Complete training statistics
- `experiment_metadata.json` - Experiment summary
"""
        
        with open(exp_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        print("🗜️  Creating zip archive...")
        
        # Create zip archive
        zip_path = results_base / f"{exp_name}_{timestamp}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in exp_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(exp_dir)
                    zf.write(file_path, arcname)
        
        zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
        
        print("✅ Experiment completed successfully!")
        print(f"📦 Archive: {zip_path} ({zip_size_mb:.1f} MB)")
        print(f"📊 Iterations: {total_iterations}")
        
        if final_reward is not None:
            print(f"🎯 Final performance: {final_reward:.4f}")
        
        return str(zip_path)
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        
        # Save error info
        error_info = {
            'error': str(e),
            'experiment_name': exp_name,
            'timestamp': timestamp,
            'config': config
        }
        
        with open(exp_dir / "error_log.json", 'w') as f:
            json.dump(error_info, f, indent=2)
        
        zip_path = results_base / f"{exp_name}_{timestamp}_FAILED.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in exp_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(exp_dir)
                    zf.write(file_path, arcname)
        
        print(f"💾 Error info saved: {zip_path}")
        raise

def run_sweep_experiments(configs, config_path):
    """Run parameter sweep"""
    print(f"🧪 Running parameter sweep: {len(configs)} configurations")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = Path(config_path).stem
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Configuration {i+1}/{len(configs)}")
        
        # Generate unique name
        config['experiment_name'] = f"{base_name}_sweep_{i+1:03d}"
        
        try:
            result_path = run_single_experiment(config, f"{config_path}_sweep_{i+1}")
            results.append({'config': config, 'result': result_path, 'status': 'success'})
        except Exception as e:
            print(f"❌ Configuration {i+1} failed: {e}")
            results.append({'config': config, 'error': str(e), 'status': 'failed'})
    
    # Create sweep summary
    sweep_path = Path("results") / f"{base_name}_sweep_{timestamp}.json"
    with open(sweep_path, 'w') as f:
        json.dump({
            'sweep_info': {
                'total_configs': len(configs),
                'successful': len([r for r in results if r['status'] == 'success']),
                'failed': len([r for r in results if r['status'] == 'failed']),
                'timestamp': timestamp
            },
            'results': results
        }, f, indent=2)
    
        # Auto-analyze results if we have enough successful experiments
    successful_count = len([r for r in results if r['status'] == 'success'])
    if successful_count >= 3:
        print(f"\n🔍 Auto-analyzing {successful_count} successful experiments...")
        try:
            from analyze_sweep_results import generate_analysis_report
            analysis_dir = sweep_path.parent / f"{base_name}_analysis_{timestamp}"
            generate_analysis_report(str(sweep_path), str(analysis_dir))
        except Exception as e:
            print(f"Auto-analysis failed: {e}")


    print(f"\n🎉 Sweep completed! Summary: {sweep_path}")
    return str(sweep_path)

def run_experiment(config_path: str):
    """Run experiment(s) - handles both single configs and sweeps"""
    
    print(f"🚀 Starting VariBAD experiment")
    print(f"Config: {config_path}")
    
    # Load config
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        base_config = json.load(f)
    
    # Parse for sweeps
    configs = parse_sweep_config(base_config)
    
    if len(configs) == 1:
        return run_single_experiment(configs[0], config_path)
    else:
        return run_sweep_experiments(configs, config_path)

def run_multiple_experiments(config_paths):
    """Run multiple experiments sequentially"""
    
    print(f"🧪 Running {len(config_paths)} experiments")
    results = []
    
    for i, config_path in enumerate(config_paths, 1):
        print(f"\n{'='*60}")
        print(f"Experiment {i}/{len(config_paths)}: {config_path}")
        print(f"{'='*60}")
        
        try:
            result_path = run_experiment(config_path)
            results.append({'config': config_path, 'result': result_path, 'status': 'success'})
        except Exception as e:
            print(f"❌ Experiment {i} failed: {e}")
            results.append({'config': config_path, 'error': str(e), 'status': 'failed'})
            continue
    
    # Summary
    print(f"\n🎉 Experiment suite completed!")
    print(f"Successful: {len([r for r in results if r['status'] == 'success'])}")
    print(f"Failed: {len([r for r in results if r['status'] == 'failed'])}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="VariBAD Portfolio Optimization Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py config/experiment1.conf
  python run_experiment.py config/experiment_sweep.conf
  python run_experiment.py config/experiment1.conf config/experiment2.conf
        """
    )
    
    parser.add_argument('configs', nargs='+', 
                       help='Path(s) to experiment config file(s)')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'],
                       help='Override device setting')
    
    args = parser.parse_args()
    
    # Validate config files
    config_files = []
    for config_pattern in args.configs:
        config_path = Path(config_pattern)
        if config_path.exists():
            config_files.append(str(config_path))
        else:
            print(f"❌ Config file not found: {config_pattern}")
            sys.exit(1)
    
    # Override device if specified
    if args.device:
        for config_file in config_files:
            with open(config_file, 'r') as f:
                config = json.load(f)
            config.setdefault('environment', {})['device'] = args.device
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
    
    # Run experiments
    if len(config_files) == 1:
        run_experiment(config_files[0])
    else:
        run_multiple_experiments(config_files)

if __name__ == "__main__":
    main()