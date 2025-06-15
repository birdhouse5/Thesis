"""Analyze experiment logs and create comparison visualizations."""

import json
import yaml
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Any


class LogAnalyzer:
    """Analyze and compare experiment logs."""
    
    def __init__(self, log_dir: str = "results/logs"):
        self.log_dir = Path(log_dir)
        self.experiments = self._load_all_experiments()
        
    def _load_all_experiments(self) -> List[Dict]:
        """Load all experiment logs."""
        experiments = []
        
        for exp_dir in self.log_dir.iterdir():
            if exp_dir.is_dir():
                json_path = exp_dir / 'experiment_log.json'
                if json_path.exists():
                    with open(json_path, 'r') as f:
                        exp_data = json.load(f)
                        exp_data['log_dir'] = exp_dir
                        experiments.append(exp_data)
        
        # Sort by timestamp
        experiments.sort(key=lambda x: x['timestamp'])
        
        return experiments
    
    def get_experiment_summary(self) -> pd.DataFrame:
        """Create summary DataFrame of all experiments."""
        summary_data = []
        
        for exp in self.experiments:
            summary = {
                'experiment_id': exp['experiment_id'],
                'timestamp': exp['timestamp'],
                'description': exp.get('description', ''),
                'duration_seconds': exp.get('duration_seconds', 0),
                'n_warnings': len(exp.get('warnings', [])),
                'n_errors': len(exp.get('errors', [])),
                'success': len(exp.get('errors', [])) == 0
            }
            
            # Add key metrics
            metrics = exp.get('metrics', {})
            for key in ['total_episodes', 'n_unique_tasks', 'train_percentage', 
                       'val_percentage', 'test_percentage']:
                summary[key] = metrics.get(key, None)
            
            # Add decision variables
            decision_vars = exp.get('decision_variables', {})
            summary['split_method'] = decision_vars.get('split_method', '')
            summary['buffer_days'] = decision_vars.get('buffer_days', 0)
            summary['episode_length'] = decision_vars.get('episode_length', 0)
            
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Compare specific experiments."""
        comparison_data = []
        
        for exp in self.experiments:
            if exp['experiment_id'] in experiment_ids:
                row = {'experiment_id': exp['experiment_id']}
                
                # Add all decision variables
                for key, value in exp.get('decision_variables', {}).items():
                    row[f'param_{key}'] = value
                
                # Add all metrics
                for key, value in exp.get('metrics', {}).items():
                    row[f'metric_{key}'] = value
                
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def plot_experiment_timeline(self):
        """Plot timeline of experiments."""
        df = self.get_experiment_summary()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], format='%Y%m%d_%H%M%S')
        
        # Plot timeline
        colors = ['green' if success else 'red' for success in df['success']]
        ax.scatter(df['datetime'], range(len(df)), c=colors, s=100, alpha=0.7)
        
        # Add experiment names
        for i, row in df.iterrows():
            ax.text(row['datetime'], i, row['experiment_id'].split('_')[0], 
                   ha='right', va='center', fontsize=8)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Experiment Index')
        ax.set_title('Experiment Timeline')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.scatter([], [], c='green', label='Success')
        ax.scatter([], [], c='red', label='Failed')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_parameter_impact(self, parameter: str, metric: str):
        """Plot impact of a parameter on a metric."""
        df = self.get_experiment_summary()
        
        # Filter experiments that have both parameter and metric
        mask = df[parameter].notna() & df[metric].notna()
        df_filtered = df[mask]
        
        if len(df_filtered) == 0:
            print(f"No experiments with both {parameter} and {metric}")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(df_filtered[parameter], df_filtered[metric], s=100, alpha=0.7)
        
        # Add labels
        for i, row in df_filtered.iterrows():
            ax.annotate(row['experiment_id'].split('_')[-1], 
                       (row[parameter], row[metric]),
                       fontsize=8, alpha=0.5)
        
        ax.set_xlabel(parameter)
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} vs {parameter}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_best_experiment(self, metric: str, minimize: bool = False) -> Dict:
        """Find best experiment based on a metric."""
        best_exp = None
        best_value = float('inf') if minimize else float('-inf')
        
        for exp in self.experiments:
            value = exp.get('metrics', {}).get(metric)
            if value is not None:
                if (minimize and value < best_value) or (not minimize and value > best_value):
                    best_value = value
                    best_exp = exp
        
        return best_exp
    
    def create_experiment_report(self, experiment_id: str, output_path: str = None):
        """Create detailed report for a specific experiment."""
        exp = None
        for e in self.experiments:
            if e['experiment_id'] == experiment_id:
                exp = e
                break
        
        if not exp:
            print(f"Experiment {experiment_id} not found")
            return
        
        report = f"""# Experiment Report: {experiment_id}

## Overview
- **Date**: {exp['timestamp']}
- **Duration**: {exp.get('duration_seconds', 0):.1f} seconds
- **Success**: {len(exp.get('errors', [])) == 0}
- **Description**: {exp.get('description', 'N/A')}

## Configuration

### Input Configuration
"""
        for key, value in exp.get('input_config', {}).items():
            report += f"- **{key}**: {value}\n"
        
        report += "\n### Decision Variables\n"
        for key, value in exp.get('decision_variables', {}).items():
            report += f"- **{key}**: {value}\n"
        
        report += "\n## Results\n\n### Metrics\n"
        for key, value in sorted(exp.get('metrics', {}).items()):
            report += f"- **{key}**: {value}\n"
        
        if exp.get('warnings'):
            report += "\n### Warnings\n"
            for warning in exp['warnings']:
                report += f"- ⚠️ {warning}\n"
        
        if exp.get('observations'):
            report += "\n## Observations\n"
            for obs in exp['observations']:
                report += f"- {obs}\n"
        
        if exp.get('next_steps'):
            report += "\n## Suggested Next Steps\n"
            for step in exp['next_steps']:
                report += f"- {step}\n"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
        
        return report


def analyze_latest_experiment():
    """Quick analysis of the latest experiment."""
    analyzer = LogAnalyzer()
    
    if not analyzer.experiments:
        print("No experiments found!")
        return
    
    # Filter for variBAD experiments
    varibad_experiments = [exp for exp in analyzer.experiments 
                          if 'varibad' in exp['experiment_id'].lower()]
    
    if varibad_experiments:
        latest = varibad_experiments[-1]
    else:
        latest = analyzer.experiments[-1]
    
    print(f"\nLatest Experiment: {latest['experiment_id']}")
    print(f"Description: {latest.get('description', 'N/A')}")
    print(f"Success: {len(latest.get('errors', [])) == 0}")
    
    print("\nKey Metrics:")
    for key, value in sorted(latest.get('metrics', {}).items()):
        print(f"  {key}: {value}")
    
    if latest.get('observations'):
        print("\nObservations:")
        for obs in latest['observations']:
            print(f"  - {obs}")
    
    if latest.get('next_steps'):
        print("\nNext Steps:")
        for step in latest['next_steps']:
            print(f"  - {step}")
    
    print(f"\nFull logs at: {latest['log_dir']}")


if __name__ == "__main__":
    analyze_latest_experiment()