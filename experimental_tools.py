#!/usr/bin/env python3
"""
Phase 3: Experimental Tools Suite
Advanced experimentation capabilities for VariBAD research
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import argparse
from experiment_database import ExperimentDatabase

class VariBADExperimentSuite:
    """Advanced experimental tools for VariBAD research"""
    
    def __init__(self):
        self.db = ExperimentDatabase()
        self.results_dir = Path("results/analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_ablation_study(self, 
                          base_config: str = "profiles/development.conf",
                          components: List[str] = None) -> str:
        """
        Run systematic ablation study
        
        Args:
            base_config: Base configuration to use
            components: Components to ablate ["short_selling", "vae", "belief"]
        """
        
        if components is None:
            components = ["short_selling", "vae_updates", "latent_dim"]
        
        print(f"🧪 Running Ablation Study")
        print(f"Base config: {base_config}")
        print(f"Components: {components}")
        
        ablation_configs = []
        
        # Generate ablation configurations
        for component in components:
            if component == "short_selling":
                config = {
                    "portfolio": {"short_selling": False},
                    "experiment": {
                        "name": f"ablation_no_short",
                        "description": f"Ablation: No short selling",
                        "tags": ["ablation", "no_short"]
                    }
                }
            elif component == "vae_updates":
                config = {
                    "training": {"vae_updates": 1},
                    "experiment": {
                        "name": f"ablation_minimal_vae",
                        "description": f"Ablation: Minimal VAE updates",
                        "tags": ["ablation", "minimal_vae"]
                    }
                }
            elif component == "latent_dim":
                config = {
                    "varibad": {"latent_dim": 1},
                    "experiment": {
                        "name": f"ablation_minimal_latent",
                        "description": f"Ablation: Minimal latent dimension",
                        "tags": ["ablation", "minimal_latent"]
                    }
                }
            
            ablation_configs.append(config)
        
        # Save ablation configs and run them
        study_name = f"ablation_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        for i, config in enumerate(ablation_configs):
            config_path = f"config/experiments/{study_name}_{i:02d}.conf"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Created ablation config: {config_path}")
        
        print(f"\nTo run ablation study:")
        for i in range(len(ablation_configs)):
            config_path = f"config/experiments/{study_name}_{i:02d}.conf"
            print(f"  python varibad/main.py --config {config_path}")
        
        return study_name
    
    def run_hyperparameter_sweep(self, 
                                param_ranges: Dict[str, List[Any]],
                                base_config: str = "profiles/development.conf",
                                max_combinations: int = 20) -> str:
        """
        Run systematic hyperparameter sweep
        
        Args:
            param_ranges: Dictionary of parameter ranges
            base_config: Base configuration
            max_combinations: Maximum number of combinations to test
        """
        
        print(f"🔬 Running Hyperparameter Sweep")
        print(f"Parameters: {param_ranges}")
        
        import itertools
        
        # Generate all combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        all_combinations = list(itertools.product(*param_values))
        
        # Limit combinations if too many
        if len(all_combinations) > max_combinations:
            print(f"Limiting to {max_combinations} out of {len(all_combinations)} combinations")
            # Use random sampling for fair coverage
            import random
            random.seed(42)
            selected_combinations = random.sample(all_combinations, max_combinations)
        else:
            selected_combinations = all_combinations
        
        study_name = f"hyperparam_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate sweep command
        sweep_str = []
        for param_name, param_range in param_ranges.items():
            values_str = ",".join(map(str, param_range))
            sweep_str.append(f"{param_name}={values_str}")
        
        sweep_command = f"python varibad/main.py --config {base_config} --sweep {' '.join(sweep_str)}"
        
        print(f"\nGenerated sweep command:")
        print(f"  {sweep_command}")
        
        print(f"\nThis will run {len(selected_combinations)} experiments")
        print(f"Study name: {study_name}")
        
        return sweep_command
    
    def analyze_experiment_results(self, experiment_ids: List[int] = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of experiment results
        
        Args:
            experiment_ids: Specific experiments to analyze (None for all completed)
        """
        
        print(f"📊 Analyzing Experiment Results")
        
        # Get experiments
        if experiment_ids:
            experiments_df = self.db.get_experiments()
            experiments_df = experiments_df[experiments_df['id'].isin(experiment_ids)]
        else:
            experiments_df = self.db.get_experiments(status='completed')
        
        if len(experiments_df) == 0:
            print("No completed experiments found")
            return {}
        
        print(f"Analyzing {len(experiments_df)} experiments")
        
        # Basic statistics
        analysis = {
            'summary': {
                'total_experiments': len(experiments_df),
                'avg_final_reward': experiments_df['final_reward'].mean(),
                'best_final_reward': experiments_df['final_reward'].max(),
                'worst_final_reward': experiments_df['final_reward'].min(),
                'std_final_reward': experiments_df['final_reward'].std()
            },
            'top_performers': experiments_df.nlargest(5, 'final_reward')[['name', 'final_reward', 'tags']].to_dict('records'),
            'experiment_count_by_tag': {}
        }
        
        # Analyze by tags
        all_tags = []
        for tags_str in experiments_df['tags'].dropna():
            if tags_str:
                all_tags.extend(tags_str.split(','))
        
        from collections import Counter
        tag_counts = Counter(all_tags)
        analysis['experiment_count_by_tag'] = dict(tag_counts)
        
        # Get hyperparameter analysis
        if experiment_ids:
            hyperparams_df = self.db.get_hyperparameter_comparison(experiment_ids)
        else:
            hyperparams_df = self.db.get_hyperparameter_comparison(experiments_df['id'].tolist())
        
        analysis['hyperparameter_impact'] = self._analyze_hyperparameter_impact(
            experiments_df, hyperparams_df
        )
        
        # Save analysis
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        analysis_path = self.results_dir / f"experiment_analysis_{timestamp}.json"
        
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"Analysis saved to: {analysis_path}")
        
        # Print summary
        print(f"\n📈 Analysis Summary:")
        print(f"  Best reward: {analysis['summary']['best_final_reward']:.4f}")
        print(f"  Average reward: {analysis['summary']['avg_final_reward']:.4f}")
        print(f"  Standard deviation: {analysis['summary']['std_final_reward']:.4f}")
        
        return analysis
    
    def _analyze_hyperparameter_impact(self, experiments_df: pd.DataFrame, 
                                     hyperparams_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze impact of different hyperparameters on performance"""
        
        impact_analysis = {}
        
        # Key hyperparameters to analyze
        key_params = [
            'varibad.latent_dim',
            'training.episode_length', 
            'training.vae_updates',
            'portfolio.short_selling'
        ]
        
        for param in key_params:
            if param in hyperparams_df.index:
                param_impact = self._calculate_parameter_impact(
                    experiments_df, hyperparams_df, param
                )
                impact_analysis[param] = param_impact
        
        return impact_analysis
    
    def _calculate_parameter_impact(self, experiments_df: pd.DataFrame,
                                  hyperparams_df: pd.DataFrame, 
                                  param_name: str) -> Dict[str, Any]:
        """Calculate the impact of a specific parameter on performance"""
        
        # Get parameter values and corresponding rewards
        param_values = []
        rewards = []
        
        for exp_name in hyperparams_df.columns:
            if param_name in hyperparams_df.index:
                param_val = hyperparams_df.loc[param_name, exp_name]
                exp_reward = experiments_df[experiments_df['name'] == exp_name]['final_reward'].iloc[0]
                
                if pd.notna(param_val) and pd.notna(exp_reward):
                    param_values.append(param_val)
                    rewards.append(float(exp_reward))
        
        if len(param_values) < 2:
            return {"insufficient_data": True}
        
        # Calculate correlation and group statistics
        try:
            # Convert to numeric if possible
            numeric_values = []
            for val in param_values:
                try:
                    numeric_values.append(float(val))
                except:
                    numeric_values.append(hash(str(val)) % 1000)  # Hash non-numeric values
            
            correlation = np.corrcoef(numeric_values, rewards)[0, 1] if len(set(numeric_values)) > 1 else 0
            
            # Group by parameter value
            value_groups = {}
            for val, reward in zip(param_values, rewards):
                if val not in value_groups:
                    value_groups[val] = []
                value_groups[val].append(reward)
            
            # Calculate statistics for each group
            group_stats = {}
            for val, group_rewards in value_groups.items():
                group_stats[str(val)] = {
                    'count': len(group_rewards),
                    'mean_reward': np.mean(group_rewards),
                    'std_reward': np.std(group_rewards),
                    'max_reward': np.max(group_rewards)
                }
            
            return {
                'correlation': float(correlation),
                'group_statistics': group_stats,
                'sample_size': len(param_values)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def create_comparison_report(self, experiment_ids: List[int], 
                               report_name: str = None) -> str:
        """
        Create comprehensive comparison report between experiments
        
        Args:
            experiment_ids: List of experiment IDs to compare
            report_name: Name for the comparison report
        """
        
        if report_name is None:
            report_name = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"📋 Creating Comparison Report: {report_name}")
        
        # Create comparison in database
        comparison_id = self.db.create_comparison(
            name=report_name,
            experiment_ids=experiment_ids,
            description=f"Comparison of {len(experiment_ids)} experiments"
        )
        
        # Get detailed data
        experiments_df = self.db.get_experiments()
        selected_experiments = experiments_df[experiments_df['id'].isin(experiment_ids)]
        
        hyperparams_df = self.db.get_hyperparameter_comparison(experiment_ids)
        training_curves_df = self.db.get_training_curves(experiment_ids)
        
        # Create visualizations
        self._create_comparison_plots(selected_experiments, training_curves_df, report_name)
        
        # Generate report
        report_path = self.results_dir / f"{report_name}_report.md"
        self._generate_comparison_markdown(
            selected_experiments, hyperparams_df, training_curves_df, report_path
        )
        
        print(f"Comparison report created:")
        print(f"  Database ID: {comparison_id}")
        print(f"  Report file: {report_path}")
        print(f"  Plots saved in: {self.results_dir}")
        
        return str(report_path)
    
    def _create_comparison_plots(self, experiments_df: pd.DataFrame, 
                               training_curves_df: pd.DataFrame, 
                               report_name: str):
        """Create comparison plots for experiments"""
        
        plt.style.use('default')
        
        # Performance comparison bar plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Final rewards comparison
        experiments_df.plot(x='name', y='final_reward', kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Final Episode Rewards Comparison')
        ax1.set_ylabel('Final Reward')
        ax1.tick_params(axis='x', rotation=45)
        
        # Training curves
        if len(training_curves_df) > 0:
            for exp_name in training_curves_df['name'].unique():
                exp_data = training_curves_df[training_curves_df['name'] == exp_name]
                ax2.plot(exp_data['iteration'], exp_data['episode_reward'], 
                        label=exp_name, alpha=0.7)
            
            ax2.set_title('Training Curves Comparison')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Episode Reward')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.results_dir / f"{report_name}_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plots saved: {plot_path}")
    
    def _generate_comparison_markdown(self, experiments_df: pd.DataFrame,
                                    hyperparams_df: pd.DataFrame,
                                    training_curves_df: pd.DataFrame,
                                    report_path: Path):
        """Generate markdown comparison report"""
        
        report_content = f"""# VariBAD Experiment Comparison Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

Total experiments compared: {len(experiments_df)}

### Performance Overview

| Experiment | Final Reward | Status | Tags |
|------------|--------------|---------|------|
"""
        
        for _, exp in experiments_df.iterrows():
            tags = exp.get('tags', '') or ''
            report_content += f"| {exp['name']} | {exp['final_reward']:.4f} | {exp['status']} | {tags} |\n"
        
        report_content += f"""

### Best Performing Experiment

**{experiments_df.loc[experiments_df['final_reward'].idxmax(), 'name']}**
- Final Reward: {experiments_df['final_reward'].max():.4f}
- Description: {experiments_df.loc[experiments_df['final_reward'].idxmax(), 'description']}

### Key Statistics

- Average Final Reward: {experiments_df['final_reward'].mean():.4f}
- Standard Deviation: {experiments_df['final_reward'].std():.4f}
- Range: {experiments_df['final_reward'].min():.4f} to {experiments_df['final_reward'].max():.4f}

## Hyperparameter Comparison

"""
        
        if len(hyperparams_df) > 0:
            # Add hyperparameter table
            report_content += "| Parameter | " + " | ".join(hyperparams_df.columns) + " |\n"
            report_content += "|-----------|" + "|".join(["-------"] * len(hyperparams_df.columns)) + "|\n"
            
            for param in hyperparams_df.index:
                row = f"| {param} |"
                for exp_name in hyperparams_df.columns:
                    value = hyperparams_df.loc[param, exp_name]
                    row += f" {value} |"
                report_content += row + "\n"
        
        report_content += f"""

## Training Progress

{len(training_curves_df)} training data points collected across all experiments.

## Recommendations

Based on this comparison:

1. **Best Configuration**: Use parameters from {experiments_df.loc[experiments_df['final_reward'].idxmax(), 'name']}
2. **Performance Spread**: {experiments_df['final_reward'].std():.4f} standard deviation suggests {'high' if experiments_df['final_reward'].std() > 0.1 else 'moderate'} sensitivity to hyperparameters
3. **Sample Size**: Consider running more experiments if standard deviation is high

## Files

- Comparison plots: `{report_path.parent.name}_comparison.png`
- Raw data available in experiment database

---
*Generated by VariBAD Experimental Tools Suite*
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    def suggest_next_experiments(self, n_suggestions: int = 5) -> List[Dict[str, Any]]:
        """
        Suggest next experiments based on current results
        
        Args:
            n_suggestions: Number of experiment suggestions to generate
        """
        
        print(f"🎯 Generating {n_suggestions} experiment suggestions")
        
        # Get current experiments
        experiments_df = self.db.get_experiments(status='completed')
        
        if len(experiments_df) == 0:
            # Default suggestions for first experiments
            return self._get_default_suggestions()
        
        # Analyze current results to suggest improvements
        suggestions = []
        
        # Suggestion 1: Improve best performer
        best_exp = experiments_df.loc[experiments_df['final_reward'].idxmax()]
        suggestions.append({
            'type': 'optimize_best',
            'description': f'Optimize around best performer: {best_exp["name"]}',
            'base_config': 'profiles/development.conf',
            'modifications': {
                'experiment': {
                    'name': 'optimize_best_performer',
                    'description': f'Fine-tune around {best_exp["name"]}',
                    'tags': ['optimization', 'best_performer']
                }
            },
            'rationale': f'Build on best result ({best_exp["final_reward"]:.4f})'
        })
        
        # Suggestion 2: Ablation study
        suggestions.append({
            'type': 'ablation',
            'description': 'Run ablation study on key components',
            'command': 'python experimental_tools.py ablation',
            'rationale': 'Understand component contributions'
        })
        
        # Suggestion 3: Parameter sweep
        suggestions.append({
            'type': 'parameter_sweep', 
            'description': 'Systematic hyperparameter exploration',
            'base_config': 'profiles/development.conf',
            'sweep_params': {
                'latent_dim': [3, 5, 8, 12],
                'episode_length': [30, 60, 90]
            },
            'rationale': 'Explore parameter space systematically'
        })
        
        # Add more suggestions as needed
        return suggestions[:n_suggestions]
    
    def _get_default_suggestions(self) -> List[Dict[str, Any]]:
        """Default suggestions when no experiments exist yet"""
        
        return [
            {
                'type': 'baseline',
                'description': 'Run baseline experiment',
                'command': 'python varibad/main.py --config experiments/exp_001_baseline.conf',
                'rationale': 'Establish baseline performance'
            },
            {
                'type': 'ablation_short',
                'description': 'Test long-only vs long-short',
                'command': 'python varibad/main.py --config experiments/exp_002_no_short.conf',
                'rationale': 'Understand short selling impact'
            },
            {
                'type': 'latent_sweep',
                'description': 'Explore belief complexity',
                'command': 'python varibad/main.py --config experiments/exp_003_latent_dim_sweep.conf',
                'rationale': 'Find optimal latent dimension'
            }
        ]


def create_experimental_cli():
    """Command-line interface for experimental tools"""
    
    parser = argparse.ArgumentParser(description="VariBAD Experimental Tools")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ablation study
    ablation_parser = subparsers.add_parser('ablation', help='Run ablation study')
    ablation_parser.add_argument('--base', default='profiles/development.conf', help='Base configuration')
    ablation_parser.add_argument('--components', nargs='+', 
                                default=['short_selling', 'vae_updates', 'latent_dim'],
                                help='Components to ablate')
    
    # Hyperparameter sweep
    sweep_parser = subparsers.add_parser('sweep', help='Generate hyperparameter sweep')
    sweep_parser.add_argument('--latent_dim', nargs='+', type=int, default=[3,5,8,12])
    sweep_parser.add_argument('--episode_length', nargs='+', type=int, default=[30,60,90])
    sweep_parser.add_argument('--base', default='profiles/development.conf')
    
    # Analysis
    analysis_parser = subparsers.add_parser('analyze', help='Analyze experiment results')
    analysis_parser.add_argument('--experiments', nargs='+', type=int, help='Specific experiment IDs')
    
    # Comparison
    compare_parser = subparsers.add_parser('compare', help='Compare experiments')
    compare_parser.add_argument('experiment_ids', nargs='+', type=int, help='Experiment IDs to compare')
    compare_parser.add_argument('--name', required=True, help='Comparison name')
    
    # Suggestions
    suggest_parser = subparsers.add_parser('suggest', help='Suggest next experiments')
    suggest_parser.add_argument('--count', type=int, default=5, help='Number of suggestions')
    
    args = parser.parse_args()
    
    # Initialize experimental suite
    suite = VariBADExperimentSuite()
    
    if args.command == 'ablation':
        study_name = suite.run_ablation_study(args.base, args.components)
        print(f"Ablation study '{study_name}' configured")
        
    elif args.command == 'sweep':
        param_ranges = {
            'latent_dim': args.latent_dim,
            'episode_length': args.episode_length
        }
        sweep_command = suite.run_hyperparameter_sweep(param_ranges, args.base)
        print(f"Run this command to execute sweep:")
        print(f"  {sweep_command}")
        
    elif args.command == 'analyze':
        analysis = suite.analyze_experiment_results(args.experiments)
        
    elif args.command == 'compare':
        report_path = suite.create_comparison_report(args.experiment_ids, args.name)
        print(f"Comparison report created: {report_path}")
        
    elif args.command == 'suggest':
        suggestions = suite.suggest_next_experiments(args.count)
        
        print(f"Experiment Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n{i}. {suggestion['description']}")
            print(f"   Type: {suggestion['type']}")
            print(f"   Rationale: {suggestion['rationale']}")
            if 'command' in suggestion:
                print(f"   Command: {suggestion['command']}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    create_experimental_cli()