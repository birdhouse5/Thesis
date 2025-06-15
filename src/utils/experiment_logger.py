"""Comprehensive experiment logging system."""

import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import traceback
import sys


@dataclass
class ExperimentLog:
    """Container for all experiment information."""
    # Metadata
    experiment_id: str
    timestamp: str
    description: str
    
    # Configuration
    input_config: Dict[str, Any]
    decision_variables: Dict[str, Any]
    
    # Data info
    data_summary: Dict[str, Any]
    
    # Results
    split_results: Dict[str, Any]
    episode_results: Dict[str, Any]
    
    # Performance metrics
    metrics: Dict[str, Any]
    
    # Errors and warnings
    warnings: List[str]
    errors: List[str]
    
    # Insights
    observations: List[str]
    next_steps: List[str]


class ExperimentLogger:
    """Comprehensive logging for experiments."""
    
    def __init__(self, experiment_name: str, log_dir: str = "results/logs"):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{self.timestamp}"
        
        # Create log directory
        self.log_dir = Path(log_dir)
        self.experiment_dir = self.log_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log data
        self.log_data = {
            'experiment_id': self.experiment_id,
            'timestamp': self.timestamp,
            'start_time': datetime.now().isoformat(),
            'description': '',
            'input_config': {},
            'decision_variables': {},
            'data_summary': {},
            'split_results': {},
            'episode_results': {},
            'metrics': {},
            'warnings': [],
            'errors': [],
            'observations': [],
            'next_steps': [],
            'checkpoints': []  # For tracking progress
        }
        
        # Setup file logging
        self._setup_file_logging()
        
        # Log experiment start
        self.logger.info(f"=" * 60)
        self.logger.info(f"EXPERIMENT: {self.experiment_name}")
        self.logger.info(f"ID: {self.experiment_id}")
        self.logger.info(f"Started: {self.log_data['start_time']}")
        self.logger.info(f"=" * 60)
        
    def _setup_file_logging(self):
        """Setup detailed file logging."""
        # Create logger
        self.logger = logging.getLogger(self.experiment_id)
        self.logger.setLevel(logging.DEBUG)
        
        # File handler for detailed logs
        fh = logging.FileHandler(self.experiment_dir / 'detailed_log.txt')
        fh.setLevel(logging.DEBUG)
        
        # Console handler for important messages
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_description(self, description: str):
        """Log experiment description."""
        self.log_data['description'] = description
        self.logger.info(f"Description: {description}")
    
    def log_config(self, config: Dict[str, Any], config_type: str = "input"):
        """Log configuration."""
        if config_type == "input":
            self.log_data['input_config'] = config
            self.logger.info("Input Configuration:")
        elif config_type == "decision":
            self.log_data['decision_variables'] = config
            self.logger.info("Decision Variables:")
        
        # Pretty print config
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_data_summary(self, data_summary: Dict[str, Any]):
        """Log data summary statistics."""
        self.log_data['data_summary'] = data_summary
        self.logger.info("\nData Summary:")
        
        for key, value in data_summary.items():
            if isinstance(value, dict):
                self.logger.info(f"  {key}:")
                for k, v in value.items():
                    self.logger.info(f"    {k}: {v}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_checkpoint(self, name: str, details: Dict[str, Any]):
        """Log a checkpoint in the experiment."""
        checkpoint = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        self.log_data['checkpoints'].append(checkpoint)
        
        self.logger.info(f"\n[CHECKPOINT] {name}")
        for key, value in details.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_split_results(self, splits: Dict[str, pd.DataFrame], 
                         summary: pd.DataFrame):
        """Log split results."""
        # Convert to serializable format
        split_info = {}
        for name, split_df in splits.items():
            if len(split_df) > 0:
                split_info[name] = {
                    'start_date': str(split_df.index[0]),
                    'end_date': str(split_df.index[-1]),
                    'n_days': len(split_df),
                    'n_assets': split_df.shape[1] if len(split_df.shape) > 1 else 1
                }
        
        self.log_data['split_results'] = {
            'splits': split_info,
            'summary': summary.to_dict() if summary is not None else {}
        }
        
        self.logger.info("\nSplit Results:")
        for name, info in split_info.items():
            self.logger.info(f"  {name}: {info['n_days']} days "
                           f"({info['start_date']} to {info['end_date']})")
    
    def log_episode_results(self, episodes: Dict[str, List[Dict]]):
        """Log episode creation results."""
        episode_summary = {}
        
        for split_name, episode_list in episodes.items():
            if episode_list:
                task_ids = [ep['task_id'] for ep in episode_list]
                unique_tasks = list(set(task_ids))
                
                episode_summary[split_name] = {
                    'n_episodes': len(episode_list),
                    'unique_tasks': unique_tasks,
                    'n_unique_tasks': len(unique_tasks),
                    'episodes_per_task': len(episode_list) / len(unique_tasks) if unique_tasks else 0
                }
        
        self.log_data['episode_results'] = episode_summary
        
        self.logger.info("\nEpisode Results:")
        for split_name, info in episode_summary.items():
            self.logger.info(f"  {split_name}:")
            self.logger.info(f"    Episodes: {info['n_episodes']}")
            self.logger.info(f"    Unique tasks: {info['n_unique_tasks']}")
            self.logger.info(f"    Tasks: {info['unique_tasks']}")
    
    def log_metric(self, name: str, value: Any):
        """Log a single metric."""
        self.log_data['metrics'][name] = value
        self.logger.info(f"Metric - {name}: {value}")
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log multiple metrics."""
        self.log_data['metrics'].update(metrics)
        self.logger.info("\nMetrics:")
        for name, value in metrics.items():
            self.logger.info(f"  {name}: {value}")
    
    def log_warning(self, message: str):
        """Log a warning."""
        self.log_data['warnings'].append(message)
        self.logger.warning(f"WARNING: {message}")
    
    def log_error(self, message: str, exception: Optional[Exception] = None):
        """Log an error."""
        error_info = {
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        if exception:
            error_info['exception'] = str(exception)
            error_info['traceback'] = traceback.format_exc()
        
        self.log_data['errors'].append(error_info)
        self.logger.error(f"ERROR: {message}")
        
        if exception:
            self.logger.error(f"Exception: {exception}")
            self.logger.debug(traceback.format_exc())
    
    def log_observation(self, observation: str):
        """Log an observation or insight."""
        self.log_data['observations'].append(observation)
        self.logger.info(f"OBSERVATION: {observation}")
    
    def log_next_step(self, next_step: str):
        """Log a suggested next step."""
        self.log_data['next_steps'].append(next_step)
        self.logger.info(f"NEXT STEP: {next_step}")
    
    def save_dataframe(self, df: pd.DataFrame, name: str):
        """Save a DataFrame to the experiment directory."""
        filepath = self.experiment_dir / f"{name}.csv"
        df.to_csv(filepath)
        self.logger.info(f"Saved DataFrame '{name}' to {filepath}")
    
    def save_figure(self, fig, name: str):
        """Save a figure to the experiment directory."""
        filepath = self.experiment_dir / f"{name}.png"
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        self.logger.info(f"Saved figure '{name}' to {filepath}")
    
    def finalize(self):
        """Finalize the experiment log."""
        self.log_data['end_time'] = datetime.now().isoformat()
        
        # Calculate duration
        start = datetime.fromisoformat(self.log_data['start_time'])
        end = datetime.fromisoformat(self.log_data['end_time'])
        duration = (end - start).total_seconds()
        self.log_data['duration_seconds'] = duration
        
        # Save JSON log
        json_path = self.experiment_dir / 'experiment_log.json'
        with open(json_path, 'w') as f:
            json.dump(self.log_data, f, indent=2, default=str)
        
        # Save YAML log (more readable)
        yaml_path = self.experiment_dir / 'experiment_log.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(self.log_data, f, default_flow_style=False)
        
        # Create summary report
        self._create_summary_report()
        
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"Experiment completed in {duration:.1f} seconds")
        self.logger.info(f"Logs saved to: {self.experiment_dir}")
        self.logger.info(f"{'=' * 60}")
    
    def _create_summary_report(self):
        """Create a markdown summary report."""
        report_path = self.experiment_dir / 'SUMMARY.md'
        
        with open(report_path, 'w') as f:
            f.write(f"# Experiment Summary: {self.experiment_id}\n\n")
            f.write(f"**Date**: {self.log_data['timestamp']}\n")
            f.write(f"**Duration**: {self.log_data['duration_seconds']:.1f} seconds\n\n")
            
            f.write(f"## Description\n{self.log_data['description']}\n\n")
            
            f.write("## Configuration\n")
            f.write("### Input Variables\n")
            for key, value in self.log_data['input_config'].items():
                f.write(f"- **{key}**: {value}\n")
            
            f.write("\n### Decision Variables\n")
            for key, value in self.log_data['decision_variables'].items():
                f.write(f"- **{key}**: {value}\n")
            
            f.write("\n## Results\n")
            if self.log_data['metrics']:
                for key, value in self.log_data['metrics'].items():
                    f.write(f"- **{key}**: {value}\n")
            
            if self.log_data['warnings']:
                f.write("\n## Warnings\n")
                for warning in self.log_data['warnings']:
                    f.write(f"- {warning}\n")
            
            if self.log_data['errors']:
                f.write("\n## Errors\n")
                for error in self.log_data['errors']:
                    f.write(f"- {error['message']}\n")
            
            if self.log_data['observations']:
                f.write("\n## Observations\n")
                for obs in self.log_data['observations']:
                    f.write(f"- {obs}\n")
            
            if self.log_data['next_steps']:
                f.write("\n## Next Steps\n")
                for step in self.log_data['next_steps']:
                    f.write(f"- {step}\n")
        
        self.logger.info(f"Summary report saved to {report_path}")


def create_experiment_logger(name: str, description: str = "") -> ExperimentLogger:
    """Convenience function to create logger."""
    logger = ExperimentLogger(name)
    if description:
        logger.log_description(description)
    return logger