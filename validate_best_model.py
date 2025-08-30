#!/usr/bin/env python3
"""
Comprehensive validation script for the best Phase 2 model.
Tests the model on validation and test sets with detailed analysis.

Usage: python validate_best_model.py
"""

import torch
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple

# Import your modules
from environments.dataset import create_split_datasets
from environments.env import MetaEnv
from models.vae import VAE
from models.policy import PortfolioPolicy
from run_logger import seed_everything

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelValidator:
    """Comprehensive model validation and analysis"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and configuration
        self.checkpoint = self._load_checkpoint()
        self.config = self._load_config()
        
        # Initialize models and data
        self.vae = None
        self.policy = None
        self.datasets = None
        self.envs = {}
        
        # Results storage
        self.results = {}
        
    def _load_checkpoint(self) -> Dict:
        """Load model checkpoint"""
        logger.info(f"Loading checkpoint from {self.model_path}")
        
        if not self.model_path.exists():
            # Try to find the best model in phase 2 results
            phase2_dir = Path("results/optuna_phase2_results")
            if phase2_dir.exists():
                best_model = phase2_dir / "best_model.pt"
                if best_model.exists():
                    self.model_path = best_model
                    logger.info(f"Found best model at {best_model}")
                else:
                    # Look for trial 46 specifically
                    trial_dirs = list(Path("results/optuna_phase2_runs").glob("trial_46_*"))
                    if trial_dirs:
                        trial_model = trial_dirs[0] / "best_model.pt"
                        if trial_model.exists():
                            self.model_path = trial_model
                            logger.info(f"Found trial 46 model at {trial_model}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        return torch.load(self.model_path, map_location=self.device, weights_only=False)
    
    def _load_config(self) -> Dict:
        """Load configuration from checkpoint or separate file"""
        if 'config' in self.checkpoint:
            config = self.checkpoint['config']
            logger.info("Loaded config from checkpoint")
        elif self.config_path and self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded config from {self.config_path}")
        else:
            # Use best trial configuration from Phase 2 results
            config = {
                # Fixed architecture (Phase 1 results)
                'latent_dim': 512,
                'hidden_dim': 1024,
                
                # Best Phase 2 parameters (Trial 46)
                'vae_lr': 0.0010748206602172,
                'policy_lr': 0.0020289998766945,
                'vae_beta': 0.0125762666385515,
                'vae_update_freq': 5,
                'seq_len': 120,
                'episodes_per_task': 3,
                'batch_size': 8192,
                'vae_batch_size': 1024,
                'ppo_epochs': 8,
                'entropy_coef': 0.0013141391952945,
                
                # Environment settings
                'data_path': "environments/data/sp500_rl_ready_cleaned.parquet",
                'train_end': '2015-12-31',
                'val_end': '2020-12-31',
                'num_assets': 30,
                'device': 'cuda',
                
                # Evaluation settings
                'val_episodes': 100,
                'test_episodes': 200,
                'seed': 42
            }
            logger.info("Using reconstructed config from best trial parameters")
        
        # Ensure required fields
        config['max_horizon'] = min(config.get('seq_len', 120) - 10, int(config.get('seq_len', 120) * 0.8))
        config['min_horizon'] = max(config['max_horizon'] - 15, config['max_horizon'] // 2)
        
        return config
    
    def initialize_models_and_data(self):
        """Initialize models and datasets"""
        logger.info("Initializing models and datasets...")
        
        # Set seed for reproducibility
        seed_everything(self.config.get('seed', 42))
        
        # Load datasets
        self.datasets = create_split_datasets(
            data_path=self.config['data_path'],
            train_end=self.config['train_end'],
            val_end=self.config['val_end']
        )
        
        # Create environments for each split
        for split_name, dataset in self.datasets.items():
            # Create dataset tensor
            window_size = min(self.config['seq_len'], len(dataset))
            features_list = []
            prices_list = []
            
            num_windows = max(1, (len(dataset) - self.config['seq_len']) // self.config['seq_len'])
            for i in range(num_windows):
                start_idx = i * self.config['seq_len']
                end_idx = start_idx + self.config['seq_len']
                
                if end_idx <= len(dataset):
                    window = dataset.get_window(start_idx, end_idx)
                    features_list.append(torch.tensor(window['features'], dtype=torch.float32))
                    prices_list.append(torch.tensor(window['raw_prices'], dtype=torch.float32))
            
            if features_list:
                all_features = torch.stack(features_list)
                all_prices = torch.stack(prices_list)
                
                dataset_tensor = {
                    'features': all_features.view(-1, self.config['num_assets'], dataset.num_features),
                    'raw_prices': all_prices.view(-1, self.config['num_assets'])
                }
                
                self.envs[split_name] = MetaEnv(
                    dataset=dataset_tensor,
                    feature_columns=dataset.feature_cols,
                    seq_len=self.config['seq_len'],
                    min_horizon=self.config['min_horizon'],
                    max_horizon=self.config['max_horizon']
                )
        
        # Get observation shape
        task = self.envs['train'].sample_task()
        self.envs['train'].set_task(task)
        initial_obs = self.envs['train'].reset()
        obs_shape = initial_obs.shape
        
        # Initialize models
        self.vae = VAE(
            obs_dim=obs_shape,
            num_assets=self.config['num_assets'],
            latent_dim=self.config['latent_dim'],
            hidden_dim=self.config['hidden_dim']
        ).to(self.device)
        
        self.policy = PortfolioPolicy(
            obs_shape=obs_shape,
            latent_dim=self.config['latent_dim'],
            num_assets=self.config['num_assets'],
            hidden_dim=self.config['hidden_dim']
        ).to(self.device)
        
        # Load model weights
        self.vae.load_state_dict(self.checkpoint['vae_state_dict'])
        self.policy.load_state_dict(self.checkpoint['policy_state_dict'])
        
        # Set to evaluation mode
        self.vae.eval()
        self.policy.eval()
        
        logger.info(f"Models initialized with observation shape: {obs_shape}")
        logger.info(f"VAE parameters: {sum(p.numel() for p in self.vae.parameters()):,}")
        logger.info(f"Policy parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
    
    def evaluate_split(self, split_name: str, num_episodes: int) -> Dict:
        """Comprehensive evaluation on a data split"""
        logger.info(f"Evaluating on {split_name} split ({num_episodes} episodes)")
        
        env = self.envs[split_name]
        episode_results = []
        portfolio_allocations = []
        wealth_curves = []
        
        with torch.no_grad():
            for episode in range(num_episodes):
                # Sample and set task
                task = env.sample_task()
                env.set_task(task)
                obs = env.reset()
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                # Episode tracking
                episode_reward = 0
                episode_length = 0
                done = False
                initial_capital = env.initial_capital
                
                # Trajectory context for VAE
                trajectory_context = {'observations': [], 'actions': [], 'rewards': []}
                
                # Episode-level tracking
                episode_allocations = []
                episode_wealth = [initial_capital]
                
                while not done:
                    # Get latent from VAE encoder
                    if len(trajectory_context['observations']) == 0:
                        latent = torch.zeros(1, self.config['latent_dim'], device=self.device)
                    else:
                        obs_seq = torch.stack(trajectory_context['observations']).unsqueeze(0)
                        action_seq = torch.stack(trajectory_context['actions']).unsqueeze(0)
                        reward_seq = torch.stack(trajectory_context['rewards']).unsqueeze(0).unsqueeze(-1)
                        
                        mu, logvar, _ = self.vae.encode(obs_seq, action_seq, reward_seq)
                        latent = self.vae.reparameterize(mu, logvar)
                    
                    # Policy action (deterministic for evaluation)
                    action, value = self.policy.act(obs_tensor, latent, deterministic=True)
                    action_cpu = action.squeeze(0).detach().cpu().numpy()
                    
                    # Store allocation
                    episode_allocations.append(action_cpu.copy())
                    
                    # Environment step
                    next_obs, reward, done, info = env.step(action_cpu)
                    episode_reward += reward
                    episode_length += 1
                    
                    # Store wealth
                    episode_wealth.append(env.current_capital)
                    
                    # Update trajectory context
                    trajectory_context['observations'].append(obs_tensor.squeeze(0).detach())
                    trajectory_context['actions'].append(action.squeeze(0).detach())
                    trajectory_context['rewards'].append(torch.tensor(reward, device=self.device))
                    
                    if not done:
                        obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                # Episode summary
                final_capital = env.current_capital
                total_return = (final_capital - initial_capital) / initial_capital
                
                # Calculate additional metrics
                allocations_array = np.array(episode_allocations)
                avg_allocation = np.mean(allocations_array, axis=0)
                portfolio_concentration = np.max(avg_allocation)
                avg_cash_position = 1.0 - np.mean(np.sum(allocations_array, axis=1))
                
                # Portfolio turnover (sum of absolute changes)
                if len(episode_allocations) > 1:
                    turnover = np.sum([np.sum(np.abs(allocations_array[i] - allocations_array[i-1])) 
                                     for i in range(1, len(allocations_array))])
                else:
                    turnover = 0.0
                
                # Drawdown calculation
                if len(episode_wealth) > 1:
                    wealth_array = np.array(episode_wealth)
                    running_max = np.maximum.accumulate(wealth_array)
                    drawdown = (wealth_array - running_max) / running_max
                    max_drawdown = np.min(drawdown)
                else:
                    max_drawdown = 0.0
                
                episode_result = {
                    'episode': episode,
                    'episode_reward': episode_reward,
                    'episode_length': episode_length,
                    'total_return': total_return,
                    'final_capital': final_capital,
                    'portfolio_concentration': portfolio_concentration,
                    'avg_cash_position': avg_cash_position,
                    'portfolio_turnover': turnover,
                    'max_drawdown': max_drawdown,
                    'task_id': getattr(env, 'task_id', None)
                }
                
                episode_results.append(episode_result)
                portfolio_allocations.append(allocations_array)
                wealth_curves.append(episode_wealth)
                
                if (episode + 1) % 20 == 0:
                    logger.info(f"  Completed {episode + 1}/{num_episodes} episodes")
        
        # Aggregate results
        results_df = pd.DataFrame(episode_results)
        
        summary = {
            'num_episodes': num_episodes,
            'avg_reward': results_df['episode_reward'].mean(),
            'std_reward': results_df['episode_reward'].std(),
            'avg_return': results_df['total_return'].mean(),
            'std_return': results_df['total_return'].std(),
            'avg_concentration': results_df['portfolio_concentration'].mean(),
            'avg_cash_position': results_df['avg_cash_position'].mean(),
            'avg_turnover': results_df['portfolio_turnover'].mean(),
            'avg_max_drawdown': results_df['max_drawdown'].mean(),
            'success_rate': (results_df['total_return'] > 0).mean(),
            'episode_results': episode_results,
            'portfolio_allocations': portfolio_allocations,
            'wealth_curves': wealth_curves
        }
        
        logger.info(f"{split_name} Results Summary:")
        logger.info(f"  Average Reward (Sharpe): {summary['avg_reward']:.4f} ± {summary['std_reward']:.4f}")
        logger.info(f"  Average Return: {summary['avg_return']:.4f} ± {summary['std_return']:.4f}")
        logger.info(f"  Average Concentration: {summary['avg_concentration']:.4f}")
        logger.info(f"  Average Cash Position: {summary['avg_cash_position']:.4f}")
        logger.info(f"  Average Turnover: {summary['avg_turnover']:.4f}")
        logger.info(f"  Success Rate: {summary['success_rate']:.2%}")
        
        return summary
    
    def compare_with_baselines(self, split_name: str) -> Dict:
        """Compare with simple baseline strategies"""
        logger.info(f"Computing baseline comparisons for {split_name}")
        
        env = self.envs[split_name]
        baselines = {}
        
        # Equal weight baseline
        equal_weight_returns = []
        for episode in range(50):  # Smaller sample for baselines
            task = env.sample_task()
            env.set_task(task)
            env.reset()
            
            initial_capital = env.initial_capital
            # Equal allocation across all assets
            equal_action = np.ones(self.config['num_assets']) / self.config['num_assets']
            
            episode_return = 0
            done = False
            while not done:
                _, reward, done, _ = env.step(equal_action)
                episode_return += reward
            
            final_capital = env.current_capital
            total_return = (final_capital - initial_capital) / initial_capital
            equal_weight_returns.append(total_return)
        
        baselines['equal_weight'] = {
            'avg_return': np.mean(equal_weight_returns),
            'std_return': np.std(equal_weight_returns),
            'sharpe_proxy': np.mean(equal_weight_returns) / (np.std(equal_weight_returns) + 1e-8)
        }
        
        # Cash-only baseline (no investment)
        baselines['cash_only'] = {
            'avg_return': 0.0,
            'std_return': 0.0,
            'sharpe_proxy': 0.0
        }
        
        logger.info("Baseline Results:")
        for name, metrics in baselines.items():
            logger.info(f"  {name}: Return={metrics['avg_return']:.4f}, Sharpe={metrics['sharpe_proxy']:.4f}")
        
        return baselines
    
    def generate_visualizations(self, results: Dict, output_dir: Path):
        """Generate comprehensive visualizations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Performance comparison across splits
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Analysis', fontsize=16)
        
        splits = ['val', 'test'] if 'test' in results else ['val']
        
        # Reward distribution
        ax = axes[0, 0]
        for split in splits:
            rewards = [ep['episode_reward'] for ep in results[split]['episode_results']]
            ax.hist(rewards, alpha=0.7, label=f'{split.title()} (μ={np.mean(rewards):.2f})', bins=20)
        ax.set_xlabel('Episode Reward (Sharpe)')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Return distribution
        ax = axes[0, 1]
        for split in splits:
            returns = [ep['total_return'] for ep in results[split]['episode_results']]
            ax.hist(returns, alpha=0.7, label=f'{split.title()} (μ={np.mean(returns):.4f})', bins=20)
        ax.set_xlabel('Total Return')
        ax.set_ylabel('Frequency')
        ax.set_title('Return Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Portfolio concentration
        ax = axes[1, 0]
        for split in splits:
            concentrations = [ep['portfolio_concentration'] for ep in results[split]['episode_results']]
            ax.hist(concentrations, alpha=0.7, label=f'{split.title()} (μ={np.mean(concentrations):.3f})', bins=20)
        ax.set_xlabel('Portfolio Concentration (Max Weight)')
        ax.set_ylabel('Frequency')
        ax.set_title('Portfolio Concentration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Turnover analysis
        ax = axes[1, 1]
        for split in splits:
            turnovers = [ep['portfolio_turnover'] for ep in results[split]['episode_results']]
            ax.hist(turnovers, alpha=0.7, label=f'{split.title()} (μ={np.mean(turnovers):.2f})', bins=20)
        ax.set_xlabel('Portfolio Turnover')
        ax.set_ylabel('Frequency')
        ax.set_title('Portfolio Turnover')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Sample wealth curves
        if 'val' in results:
            fig, ax = plt.subplots(figsize=(12, 6))
            wealth_curves = results['val']['wealth_curves'][:10]  # First 10 episodes
            
            for i, curve in enumerate(wealth_curves):
                steps = np.arange(len(curve))
                ax.plot(steps, curve, alpha=0.7, label=f'Episode {i+1}' if i < 5 else None)
            
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Portfolio Value ($)')
            ax.set_title('Sample Wealth Curves (Validation)')
            ax.grid(True, alpha=0.3)
            if len(wealth_curves) <= 5:
                ax.legend()
            
            plt.tight_layout()
            plt.savefig(output_dir / 'wealth_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Average portfolio allocation
        if 'val' in results:
            allocations = results['val']['portfolio_allocations']
            if allocations:
                # Compute average allocation across all episodes
                all_allocations = np.concatenate(allocations, axis=0)
                avg_allocation = np.mean(all_allocations, axis=0)
                
                # Get ticker names if available
                tickers = getattr(self.datasets['val'], 'tickers', [f'Asset_{i}' for i in range(len(avg_allocation))])
                
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(range(len(avg_allocation)), avg_allocation)
                ax.set_xlabel('Assets')
                ax.set_ylabel('Average Allocation')
                ax.set_title('Average Portfolio Allocation')
                ax.set_xticks(range(len(avg_allocation)))
                ax.set_xticklabels(tickers, rotation=45, ha='right')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar, val in zip(bars, avg_allocation):
                    if val > 0.01:  # Only label significant allocations
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                               f'{val:.3f}', ha='center', va='bottom', fontsize=8)
                
                plt.tight_layout()
                plt.savefig(output_dir / 'portfolio_allocation.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def run_comprehensive_validation(self) -> Dict:
        """Run complete validation pipeline"""
        logger.info("="*60)
        logger.info("COMPREHENSIVE MODEL VALIDATION")
        logger.info("="*60)
        
        # Initialize everything
        self.initialize_models_and_data()
        
        # Model info
        logger.info(f"Model loaded from: {self.model_path}")
        if 'trial_number' in self.checkpoint:
            logger.info(f"Trial number: {self.checkpoint['trial_number']}")
        if 'best_val_sharpe' in self.checkpoint:
            logger.info(f"Training best validation Sharpe: {self.checkpoint['best_val_sharpe']:.4f}")
        
        # Run evaluations
        results = {}
        
        # Validation set
        results['val'] = self.evaluate_split('val', self.config['val_episodes'])
        
        # Test set (if requested)
        if 'test_episodes' in self.config and self.config['test_episodes'] > 0:
            results['test'] = self.evaluate_split('test', self.config['test_episodes'])
        
        # Baseline comparisons
        results['baselines'] = self.compare_with_baselines('val')
        
        # Create output directory
        output_dir = Path("validation_results") / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations
        self.generate_visualizations(results, output_dir)
        
        # Save detailed results
        summary_file = output_dir / "validation_summary.json"
        summary_data = {
            'model_path': str(self.model_path),
            'config': self.config,
            'checkpoint_info': {
                'trial_number': self.checkpoint.get('trial_number'),
                'training_best_val_sharpe': self.checkpoint.get('best_val_sharpe'),
                'episodes_trained': self.checkpoint.get('episodes_trained')
            },
            'validation_results': {
                split: {k: v for k, v in results[split].items() 
                       if k not in ['episode_results', 'portfolio_allocations', 'wealth_curves']}
                for split in results if split != 'baselines'
            },
            'baseline_results': results['baselines']
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        # Print final summary
        logger.info("="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)
        
        for split in ['val', 'test']:
            if split in results:
                r = results[split]
                logger.info(f"\n{split.upper()} SET RESULTS:")
                logger.info(f"  Episodes: {r['num_episodes']}")
                logger.info(f"  Avg Sharpe: {r['avg_reward']:.4f} ± {r['std_reward']:.4f}")
                logger.info(f"  Avg Return: {r['avg_return']:.4f} ± {r['std_return']:.4f}")
                logger.info(f"  Success Rate: {r['success_rate']:.2%}")
                logger.info(f"  Avg Concentration: {r['avg_concentration']:.4f}")
                logger.info(f"  Avg Turnover: {r['avg_turnover']:.4f}")
        
        logger.info(f"\nBASELINE COMPARISONS:")
        for name, metrics in results['baselines'].items():
            logger.info(f"  {name}: Return={metrics['avg_return']:.4f}, Sharpe={metrics['sharpe_proxy']:.4f}")
        
        if 'val' in results:
            val_sharpe = results['val']['avg_reward']
            baseline_sharpe = results['baselines']['equal_weight']['sharpe_proxy']
            improvement = ((val_sharpe - baseline_sharpe) / abs(baseline_sharpe + 1e-8)) * 100 if baseline_sharpe != 0 else float('inf')
            logger.info(f"\nIMPROVEMENT vs EQUAL WEIGHT: {improvement:+.1f}%")
        
        logger.info(f"\nDETAILED RESULTS SAVED TO: {output_dir}")
        
        self.results = results
        return results


def main():
    """Main validation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate best Phase 2 model")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, help="Path to config file")
    parser.add_argument("--val_episodes", type=int, default=100, help="Number of validation episodes")
    parser.add_argument("--test_episodes", type=int, default=200, help="Number of test episodes")
    parser.add_argument("--skip_test", action="store_true", help="Skip test set evaluation")
    
    args = parser.parse_args()
    
    # Auto-detect model path if not provided
    if not args.model_path:
        # Look for Phase 2 results
        possible_paths = [
            "results/optuna_phase2_results/best_model.pt",
            "optuna_phase2_results/best_model.pt"
        ]
        
        # Look for trial 46 specifically
        trial_dirs = list(Path("results/optuna_phase2_runs").glob("trial_46_*"))
        if trial_dirs:
            possible_paths.append(str(trial_dirs[0] / "best_model.pt"))
        
        for path in possible_paths:
            if Path(path).exists():
                args.model_path = path
                break
        
        if not args.model_path:
            logger.error("No model path provided and could not auto-detect. Please specify --model_path")
            return
    
    # Create validator
    validator = ModelValidator(
        model_path=args.model_path,
        config_path=args.config_path
    )
    
    # Override config with command line args
    if hasattr(validator, 'config'):
        validator.config['val_episodes'] = args.val_episodes
        validator.config['test_episodes'] = 0 if args.skip_test else args.test_episodes
    
    # Run validation
    try:
        results = validator.run_comprehensive_validation()
        logger.info("Validation completed successfully!")
        
        # Quick results preview
        if 'val' in results:
            val_sharpe = results['val']['avg_reward']
            if val_sharpe > 2.0:
                logger.warning("⚠️  Very high Sharpe ratio - please verify results for realism")
            elif val_sharpe > 1.0:
                logger.info("✅ Excellent performance!")
            elif val_sharpe > 0.5:
                logger.info("✅ Good performance")
            else:
                logger.info("⚠️  Performance may need improvement")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()