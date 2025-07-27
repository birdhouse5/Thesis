#!/usr/bin/env python3
"""
Enhanced VariBAD Main Script with Configuration System
Phase 2: Multi-experiment management and parameter sweeps
"""

import argparse
import os
import sys
import logging
import json
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiment_database import ExperimentDatabase

class ConfigManager:
    """Enhanced configuration management for VariBAD experiments"""
    
    def __init__(self):
        self.base_config = None
        self.current_config = None
        
    def load_base_config(self) -> Dict[str, Any]:
        """Load the base configuration"""
        base_path = Path("config/base.conf")
        if not base_path.exists():
            raise FileNotFoundError(f"Base configuration not found: {base_path}")
        
        with open(base_path, 'r') as f:
            self.base_config = json.load(f)
        
        return self.base_config
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration with inheritance from base"""
        
        # Start with base config
        if self.base_config is None:
            self.load_base_config()
        
        config = self.base_config.copy()
        
        # Load specific config if provided
        if config_path:
            config_file = Path(config_path)
            if not config_file.exists():
                # Try adding .conf extension
                config_file = Path(f"{config_path}.conf")
            if not config_file.exists():
                # Try in config directory
                config_file = Path(f"config/{config_path}")
            if not config_file.exists():
                config_file = Path(f"config/{config_path}.conf")
            
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_file, 'r') as f:
                override_config = json.load(f)
            
            # Merge configurations (deep merge)
            config = self._deep_merge(config, override_config)
        
        self.current_config = config
        return config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def override_with_args(self, config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
        """Override config with command line arguments"""
        
        # Training parameters
        if hasattr(args, 'num_iterations') and args.num_iterations is not None:
            config['training']['num_iterations'] = args.num_iterations
        if hasattr(args, 'episode_length') and args.episode_length is not None:
            config['training']['episode_length'] = args.episode_length
        if hasattr(args, 'episodes_per_iteration') and args.episodes_per_iteration is not None:
            config['training']['episodes_per_iteration'] = args.episodes_per_iteration
        if hasattr(args, 'vae_updates') and args.vae_updates is not None:
            config['training']['vae_updates'] = args.vae_updates
        
        # VariBAD parameters
        if hasattr(args, 'latent_dim') and args.latent_dim is not None:
            config['varibad']['latent_dim'] = args.latent_dim
        
        # Portfolio parameters
        if hasattr(args, 'short_selling') and args.short_selling is not None:
            config['portfolio']['short_selling'] = args.short_selling
        
        # Device
        if hasattr(args, 'device') and args.device is not None:
            config['environment']['device'] = args.device
        
        # Experiment name override
        if hasattr(args, 'name') and args.name is not None:
            config['experiment']['name'] = args.name
        
        return config
    
    def parse_sweep_parameters(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse SWEEP parameters and generate all combinations"""
        
        sweep_params = {}
        
        # Find all SWEEP parameters
        def find_sweeps(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if isinstance(value, str) and value.startswith("SWEEP:"):
                        # Parse sweep values
                        sweep_str = value[6:]  # Remove "SWEEP:"
                        try:
                            sweep_values = json.loads(sweep_str)
                            sweep_params[current_path] = sweep_values
                        except json.JSONDecodeError:
                            print(f"Warning: Invalid sweep format for {current_path}: {value}")
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
                # Set the parameter value
                self._set_nested_value(new_config, param_name, value)
            
            configs.append(new_config)
        
        return configs
    
    def _set_nested_value(self, obj: Dict, path: str, value: Any):
        """Set a nested dictionary value using dot notation"""
        keys = path.split('.')
        current = obj
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def generate_experiment_name(self, config: Dict[str, Any], args: argparse.Namespace = None) -> str:
        """Generate automatic experiment name with key parameters"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Base name from config
        base_name = config.get('experiment', {}).get('name', 'exp')
        
        # Add key parameters
        params = []
        
        # Latent dimension
        latent_dim = config.get('varibad', {}).get('latent_dim', 5)
        params.append(f"latent{latent_dim}")
        
        # Episode length if non-standard
        episode_length = config.get('training', {}).get('episode_length', 30)
        if episode_length != 30:
            params.append(f"ep{episode_length}")
        
        # Iterations if non-standard
        num_iterations = config.get('training', {}).get('num_iterations', 1000)
        if num_iterations != 1000:
            params.append(f"iter{num_iterations}")
        
        # Short selling
        short_selling = config.get('portfolio', {}).get('short_selling', True)
        if not short_selling:
            params.append("longOnly")
        
        # Combine parts
        if params:
            name = f"{base_name}_{timestamp}_{'_'.join(params)}"
        else:
            name = f"{base_name}_{timestamp}"
        
        return name


class ExperimentRunner:
    """Run experiments with configuration management"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.results_db = []
        self.db = ExperimentDatabase()
        
    def run_single_experiment(self, config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
        """Run a single experiment with the given configuration"""
        
        print(f"\n🚀 Starting experiment: {experiment_name}")
        print("=" * 60)
        
        # Register experiment in database
        exp_id = self.db.register_experiment(config, experiment_name)
        self.db.start_experiment(exp_id)        

        try:
            # Import training components
            from varibad.core.trainer import VariBADTrainer
            
            # Extract parameters from config
            training_params = config.get('training', {})
            varibad_params = config.get('varibad', {})
            portfolio_params = config.get('portfolio', {})
            env_params = config.get('environment', {})
            lr_params = config.get('learning_rates', {})
            
            # Initialize trainer
            trainer = VariBADTrainer(
                data_path=env_params.get('data_path', 'data/sp500_rl_ready_cleaned.parquet'),
                episode_length=training_params.get('episode_length', 30),
                action_dim=30,  # 30 S&P 500 companies
                latent_dim=varibad_params.get('latent_dim', 5),
                max_episodes_buffer=training_params.get('buffer_size', 500),
                enable_short_selling=portfolio_params.get('short_selling', True),
                policy_lr=lr_params.get('policy_lr', 1e-4),
                vae_encoder_lr=lr_params.get('vae_encoder_lr', 1e-4),
                vae_decoder_lr=lr_params.get('vae_decoder_lr', 1e-4),
                device=env_params.get('device', 'auto')
            )
            
            print(f"✓ Trainer initialized")
            print(f"  State dimension: {trainer.state_dim}")
            print(f"  Action dimension: {trainer.actual_action_dim}")
            print(f"  Model parameters: {sum(p.numel() for p in trainer.varibad.parameters()):,}")
            
            # Run training
            stats = trainer.train(
                num_iterations=training_params.get('num_iterations', 1000),
                episodes_per_iteration=training_params.get('episodes_per_iteration', 5),
                vae_updates_per_iteration=training_params.get('vae_updates', 10),
                eval_frequency=training_params.get('eval_frequency', 50),
                save_frequency=training_params.get('save_frequency', 100)
            )
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save model checkpoint
            checkpoint_path = f"checkpoints/{experiment_name}_{timestamp}.pt"
            trainer.save_checkpoint(checkpoint_path)
            
            # Save training statistics
            stats_path = f"results/experiments/{experiment_name}_{timestamp}.json"
            os.makedirs("results/experiments", exist_ok=True)
            
            # Prepare results
            experiment_result = {
                'experiment_name': experiment_name,
                'timestamp': timestamp,
                'config': config,
                'stats': stats,
                'checkpoint_path': checkpoint_path,
                'stats_path': stats_path,
                'status': 'completed'
            }
            
            # Convert numpy arrays to lists for JSON serialization
            json_stats = {}
            for key, value in stats.items():
                if hasattr(value, 'tolist'):
                    json_stats[key] = value.tolist()
                elif isinstance(value, (list, tuple)):
                    json_stats[key] = list(value)
                else:
                    json_stats[key] = value
            
            with open(stats_path, 'w') as f:
                json.dump({
                    'experiment': experiment_result,
                    'training_stats': json_stats
                }, f, indent=2)
            
            print(f"✅ Experiment completed successfully!")
            print(f"📊 Results saved to: {stats_path}")
            print(f"💾 Model saved to: {checkpoint_path}")
            
            # Calculate summary metrics
            if 'avg_episode_reward' in stats and stats['avg_episode_reward']:
                final_reward = stats['avg_episode_reward'][-1] if stats['avg_episode_reward'] else 0
                avg_reward = sum(stats['avg_episode_reward']) / len(stats['avg_episode_reward']) if stats['avg_episode_reward'] else 0
                
                print(f"📈 Performance Summary:")
                print(f"  Final episode reward: {final_reward:.4f}")
                print(f"  Average episode reward: {avg_reward:.4f}")
                
                experiment_result['final_reward'] = final_reward
                experiment_result['avg_reward'] = avg_reward
            
            self.db.complete_experiment(
                exp_id,
                checkpoint_path=checkpoint_path,
                results_path=stats_path,
                final_reward=experiment_result.get('final_reward'),
                avg_reward=experiment_result.get('avg_reward')
                )

            return experiment_result
            
        except Exception as e:
            self.db.fail_experiment(exp_id, str(e))
            print(f"❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'experiment_name': experiment_name,
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'config': config,
                'status': 'failed',
                'error': str(e)
            }
    
    def run_experiment_suite(self, configs: List[Dict[str, Any]], base_name: str = None) -> List[Dict[str, Any]]:
        """Run a suite of experiments (e.g., parameter sweep)"""
        
        print(f"\n🧪 Running experiment suite with {len(configs)} configurations")
        print("=" * 80)
        
        results = []
        
        for i, config in enumerate(configs):
            print(f"\n📋 Configuration {i+1}/{len(configs)}")
            
            # Generate experiment name
            if base_name:
                experiment_name = f"{base_name}_{i+1:03d}"
            else:
                experiment_name = self.config_manager.generate_experiment_name(config)
            
            # Update config with experiment name
            config['experiment']['name'] = experiment_name
            
            # Run experiment
            result = self.run_single_experiment(config, experiment_name)
            results.append(result)
            
            # Save intermediate results
            suite_results_path = f"results/experiments/suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(suite_results_path, 'w') as f:
                json.dump({
                    'suite_info': {
                        'total_experiments': len(configs),
                        'completed': i + 1,
                        'timestamp': datetime.now().isoformat()
                    },
                    'results': results
                }, f, indent=2)
        
        print(f"\n🎉 Experiment suite completed!")
        print(f"📊 Suite results saved to: {suite_results_path}")
        
        # Generate summary
        successful = [r for r in results if r['status'] == 'completed']
        failed = [r for r in results if r['status'] == 'failed']
        
        print(f"\n📈 Suite Summary:")
        print(f"  Total experiments: {len(results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        
        if successful:
            avg_final_rewards = [r.get('final_reward', 0) for r in successful if 'final_reward' in r]
            if avg_final_rewards:
                best_reward = max(avg_final_rewards)
                avg_reward = sum(avg_final_rewards) / len(avg_final_rewards)
                print(f"  Best final reward: {best_reward:.4f}")
                print(f"  Average final reward: {avg_reward:.4f}")
        
        return results


def setup_logging(log_level="INFO"):
    """Setup comprehensive logging for the pipeline - WINDOWS ENCODING FIXED"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"varibad_enhanced_{timestamp}.log"
    
    # FIXED: Use UTF-8 encoding for file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler(sys.stdout)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[file_handler, console_handler]
    )
    
    logger = logging.getLogger(__name__)
    # FIXED: Remove Unicode symbols for Windows compatibility
    logger.info(f"Enhanced VariBAD Pipeline started - logs saved to {log_file}")
    logger.info(f"Running from: {os.getcwd()}")
    return logger


def create_parser():
    """Create enhanced argument parser with configuration support"""
    
    parser = argparse.ArgumentParser(
        description="Enhanced VariBAD Portfolio Optimization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phase 2 Enhanced Examples:

Profile-based Training:
  python varibad/main.py --config profiles/debug.conf
  python varibad/main.py --config profiles/development.conf
  python varibad/main.py --config profiles/production.conf

Experiment Execution:
  python varibad/main.py --config experiments/exp_001_baseline.conf
  python varibad/main.py --config experiments/exp_002_no_short.conf

Parameter Sweeps:
  python varibad/main.py --sweep latent_dim=3,5,8,12
  python varibad/main.py --sweep episode_length=30,60,90 vae_updates=5,10,15
  python varibad/main.py --config profiles/development.conf --sweep latent_dim=5,8

Automatic Naming:
  python varibad/main.py --config profiles/debug.conf --name my_test
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train',
                      choices=['data_only', 'train', 'resume', 'evaluate', 'sweep'],
                      help='Pipeline mode to run (default: train)')
    
    # Configuration system
    parser.add_argument('--config', type=str,
                      help='Configuration file (profiles/debug.conf, experiments/exp_001.conf, etc.)')
    parser.add_argument('--base', type=str, default='config/base.conf',
                      help='Base configuration file (default: config/base.conf)')
    
    # Parameter sweeps
    parser.add_argument('--sweep', nargs='+', 
                      help='Parameter sweep: latent_dim=3,5,8 episode_length=30,60')
    
    # Experiment management
    parser.add_argument('--name', type=str,
                      help='Experiment name (overrides config)')
    parser.add_argument('--tags', nargs='+',
                      help='Experiment tags for organization')
    parser.add_argument('--description', type=str,
                      help='Experiment description')
    
    # Training parameters (can override config)
    parser.add_argument('--num_iterations', type=int,
                      help='Number of training iterations')
    parser.add_argument('--episode_length', type=int,
                      help='Length of each trading episode')
    parser.add_argument('--episodes_per_iteration', type=int,
                      help='Episodes to collect per iteration')
    parser.add_argument('--vae_updates', type=int,
                      help='VAE updates per iteration')
    parser.add_argument('--latent_dim', type=int,
                      help='Latent dimension for VariBAD')
    
    # Portfolio parameters
    parser.add_argument('--short_selling', action='store_true',
                      help='Enable short selling')
    parser.add_argument('--no_short_selling', dest='short_selling', action='store_false',
                      help='Disable short selling')
    parser.set_defaults(short_selling=None)
    
    # Environment
    parser.add_argument('--device', type=str,
                      choices=['auto', 'cpu', 'cuda'],
                      help='Device to use')
    parser.add_argument('--data_path', type=str,
                      help='Path to processed dataset')
    
    # Evaluation
    parser.add_argument('--checkpoint', type=str,
                      help='Path to checkpoint for resume/evaluate modes')
    
    # Logging
    parser.add_argument('--log_level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      help='Logging level (default: INFO)')
    
    return parser


def parse_sweep_arguments(sweep_args: List[str]) -> Dict[str, List[Any]]:
    """Parse parameter sweep arguments from command line"""
    
    sweep_params = {}
    
    for arg in sweep_args:
        if '=' not in arg:
            print(f"Warning: Invalid sweep format '{arg}'. Use param=val1,val2,val3")
            continue
        
        param_name, values_str = arg.split('=', 1)
        
        try:
            # Try to parse as JSON first
            if values_str.startswith('[') and values_str.endswith(']'):
                values = json.loads(values_str)
            else:
                # Parse comma-separated values
                values = []
                for val_str in values_str.split(','):
                    val_str = val_str.strip()
                    # Try to convert to appropriate type
                    try:
                        if val_str.lower() in ['true', 'false']:
                            values.append(val_str.lower() == 'true')
                        elif '.' in val_str:
                            values.append(float(val_str))
                        else:
                            values.append(int(val_str))
                    except ValueError:
                        values.append(val_str)  # Keep as string
            
            sweep_params[param_name] = values
            
        except Exception as e:
            print(f"Warning: Could not parse sweep values for {param_name}: {e}")
    
    return sweep_params


def apply_sweep_to_config(config: Dict[str, Any], sweep_params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Apply parameter sweep to configuration"""
    
    if not sweep_params:
        return [config]
    
    # Generate all combinations
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    
    configs = []
    for combination in itertools.product(*param_values):
        new_config = json.loads(json.dumps(config))  # Deep copy
        
        for param_name, value in zip(param_names, combination):
            # Map parameter names to config structure
            if param_name == 'latent_dim':
                new_config['varibad']['latent_dim'] = value
            elif param_name == 'episode_length':
                new_config['training']['episode_length'] = value
            elif param_name == 'episodes_per_iteration':
                new_config['training']['episodes_per_iteration'] = value
            elif param_name == 'vae_updates':
                new_config['training']['vae_updates'] = value
            elif param_name == 'short_selling':
                new_config['portfolio']['short_selling'] = value
            elif param_name == 'policy_lr':
                new_config['learning_rates']['policy_lr'] = value
            elif param_name == 'vae_encoder_lr':
                new_config['learning_rates']['vae_encoder_lr'] = value
            elif param_name == 'vae_decoder_lr':
                new_config['learning_rates']['vae_decoder_lr'] = value
            elif param_name == 'num_iterations':
                new_config['training']['num_iterations'] = value
            else:
                print(f"Warning: Unknown sweep parameter '{param_name}' - ignoring")
        
        configs.append(new_config)
    
    return configs


def main():
    """Enhanced main function with configuration management"""
    
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    try:
        # Initialize configuration manager
        config_manager = ConfigManager()
        
        # Load configuration
        if args.config:
            config = config_manager.load_config(args.config)
            logger.info(f"✓ Loaded configuration: {args.config}")
        else:
            config = config_manager.load_base_config()
            logger.info("✓ Using base configuration")
        
        # Override with command line arguments
        config = config_manager.override_with_args(config, args)
        
        # Auto-detect device if needed
        if config['environment']['device'] == 'auto':
            try:
                import torch
                config['environment']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"Auto-detected device: {config['environment']['device']}")
            except ImportError:
                config['environment']['device'] = 'cpu'
                logger.warning("PyTorch not found, using CPU")
        
        logger.info(f"🚀 Starting Enhanced VariBAD Pipeline")
        logger.info(f"Mode: {args.mode}")
        
        # Handle different modes
        if args.mode == 'data_only':
            logger.info("📊 Data processing mode")
            from varibad.data_pipeline import create_rl_dataset
            final_data_path = create_rl_dataset()
            logger.info(f"✅ Data processing complete! Dataset ready at: {final_data_path}")
        
        elif args.mode == 'train' or args.mode == 'sweep':
            logger.info("🏋️ Training mode")
            
            # Check for parameter sweeps
            configs_to_run = []
            
            # Handle config-based sweeps (SWEEP: syntax)
            if args.config:
                config_sweeps = config_manager.parse_sweep_parameters(config)
                configs_to_run.extend(config_sweeps)
            else:
                configs_to_run.append(config)
            
            # Handle command-line sweeps
            if args.sweep:
                sweep_params = parse_sweep_arguments(args.sweep)
                if sweep_params:
                    logger.info(f"🧪 Parameter sweep detected: {sweep_params}")
                    
                    # Apply sweep to all configs
                    swept_configs = []
                    for cfg in configs_to_run:
                        swept_configs.extend(apply_sweep_to_config(cfg, sweep_params))
                    configs_to_run = swept_configs
            
            # Initialize experiment runner
            runner = ExperimentRunner(config_manager)
            
            if len(configs_to_run) == 1:
                # Single experiment
                config = configs_to_run[0]
                experiment_name = config_manager.generate_experiment_name(config, args)
                if args.name:
                    experiment_name = args.name
                
                result = runner.run_single_experiment(config, experiment_name)
                
            else:
                # Experiment suite
                logger.info(f"🧪 Running experiment suite with {len(configs_to_run)} configurations")
                base_name = args.name or config.get('experiment', {}).get('name', 'sweep')
                results = runner.run_experiment_suite(configs_to_run, base_name)
        
        elif args.mode == 'evaluate':
            logger.info("📊 Evaluation mode")
            if not args.checkpoint:
                logger.error("--checkpoint required for evaluate mode")
                sys.exit(1)
            
            # Evaluation logic (implement as needed)
            logger.info("Evaluation functionality will be implemented")
        
        elif args.mode == 'resume':
            logger.info("🔄 Resume mode")
            if not args.checkpoint:
                logger.error("--checkpoint required for resume mode")
                sys.exit(1)
            
            # Resume logic (implement as needed)
            logger.info("Resume functionality will be implemented")
    
    except KeyboardInterrupt:
        logger.info("⏹️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    logger.info("🎉 Enhanced Pipeline completed successfully!")


if __name__ == "__main__":
    main()