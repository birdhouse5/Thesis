import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import json

class CSVLogger:
    """Enhanced logging system that exports comprehensive CSV files for analysis"""
    
    def __init__(self, experiment_dir):
        self.experiment_dir = Path(experiment_dir)
        
        # Create subdirectories
        self.summary_dir = self.experiment_dir / "summary"
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.training_data = []
        self.validation_data = []
        self.test_data = []
        self.episode_data = []
        self.config_data = {}
        
        # File paths in summary directory
        self.training_csv = self.summary_dir / "training_metrics.csv"
        self.validation_csv = self.summary_dir / "validation_metrics.csv"
        self.test_csv = self.summary_dir / "test_metrics.csv"
        self.episodes_csv = self.summary_dir / "episode_details.csv"
        self.config_json = self.summary_dir / "config.json"
        self.summary_csv = self.summary_dir / "experiment_summary.csv"
        
        print(f"CSV Logger initialized: {self.experiment_dir}")
        print(f"  Summary files: {self.summary_dir}")
        print(f"  Checkpoints: {self.checkpoints_dir}")
    
    def log_config(self, config):
        """Save experiment configuration"""
        # Convert config object to dict if needed
        if hasattr(config, '__dict__'):
            config_dict = {k: v for k, v in config.__dict__.items() 
                          if not k.startswith('_') and not callable(v)}
        else:
            config_dict = config
            
        # Handle non-serializable objects
        serializable_config = {}
        for k, v in config_dict.items():
            try:
                json.dumps(v)  # Test if serializable
                serializable_config[k] = v
            except (TypeError, ValueError):
                serializable_config[k] = str(v)
        
        # Add metadata
        serializable_config['experiment_dir'] = str(self.experiment_dir)
        serializable_config['start_time'] = datetime.now().isoformat()
        
        self.config_data = serializable_config
        
        # Save to JSON
        with open(self.config_json, 'w') as f:
            json.dump(serializable_config, f, indent=2)
    
    def log_training_step(self, episode, step_data):
        """Log training step data"""
        row = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'split': 'train',
            **step_data
        }
        self.training_data.append(row)
    
    def log_validation(self, episode, val_results):
        """Log validation results"""
        row = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'split': 'validation',
            **val_results
        }
        self.validation_data.append(row)
    
    def log_test(self, test_results):
        """Log final test results"""
        row = {
            'timestamp': datetime.now().isoformat(),
            'split': 'test',
            **test_results
        }
        self.test_data.append(row)
    
    def log_episode_details(self, episode, episode_info):
        """Log detailed episode information"""
        row = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            **episode_info
        }
        self.episode_data.append(row)
    
    def save_all_csvs(self):
        """Save all collected data to CSV files"""
        # Training metrics
        if self.training_data:
            df_train = pd.DataFrame(self.training_data)
            df_train.to_csv(self.training_csv, index=False)
            print(f"Saved training metrics: {len(df_train)} rows -> {self.training_csv}")
        
        # Validation metrics
        if self.validation_data:
            df_val = pd.DataFrame(self.validation_data)
            df_val.to_csv(self.validation_csv, index=False)
            print(f"Saved validation metrics: {len(df_val)} rows -> {self.validation_csv}")
        
        # Test metrics
        if self.test_data:
            df_test = pd.DataFrame(self.test_data)
            df_test.to_csv(self.test_csv, index=False)
            print(f"Saved test metrics: {len(df_test)} rows -> {self.test_csv}")
        
        # Episode details
        if self.episode_data:
            df_episodes = pd.DataFrame(self.episode_data)
            df_episodes.to_csv(self.episodes_csv, index=False)
            print(f"Saved episode details: {len(df_episodes)} rows -> {self.episodes_csv}")
        
        # Create experiment summary
        self._create_experiment_summary()
    
    def _create_experiment_summary(self):
        """Create a summary CSV with key metrics"""
        summary = {}
        
        # Add config info
        summary.update({f"config_{k}": v for k, v in self.config_data.items()})
        
        # Training summary
        if self.training_data:
            df_train = pd.DataFrame(self.training_data)
            if 'episode_reward' in df_train.columns:
                summary.update({
                    'final_training_reward': df_train['episode_reward'].iloc[-1] if len(df_train) > 0 else None,
                    'avg_training_reward': df_train['episode_reward'].mean(),
                    'best_training_reward': df_train['episode_reward'].max(),
                    'total_episodes': len(df_train)
                })
        
        # Validation summary
        if self.validation_data:
            df_val = pd.DataFrame(self.validation_data)
            if 'avg_reward' in df_val.columns:
                summary.update({
                    'best_validation_reward': df_val['avg_reward'].max(),
                    'final_validation_reward': df_val['avg_reward'].iloc[-1] if len(df_val) > 0 else None,
                    'validation_count': len(df_val)
                })
        
        # Test summary
        if self.test_data:
            df_test = pd.DataFrame(self.test_data)
            if 'avg_reward' in df_test.columns:
                summary.update({
                    'test_avg_reward': df_test['avg_reward'].iloc[0] if len(df_test) > 0 else None,
                    'test_avg_return': df_test.get('avg_return', [None]).iloc[0] if len(df_test) > 0 else None
                })
        
        # Episode details summary
        if self.episode_data:
            df_episodes = pd.DataFrame(self.episode_data)
            if 'cumulative_return' in df_episodes.columns:
                summary.update({
                    'final_cumulative_return': df_episodes['cumulative_return'].iloc[-1] if len(df_episodes) > 0 else None,
                    'avg_episode_length': df_episodes.get('episode_length', []).mean() if 'episode_length' in df_episodes.columns else None
                })
        
        # Add experiment metadata
        summary.update({
            'experiment_end_time': datetime.now().isoformat(),
            'total_training_time_minutes': None,  # Could calculate if start time stored
            'experiment_dir': str(self.experiment_dir)
        })
        
        # Save summary
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(self.summary_csv, index=False)
        print(f"Saved experiment summary -> {self.summary_csv}")
    
    def get_training_curve_data(self):
        """Return training curve data for plotting"""
        if not self.training_data:
            return None
        
        df = pd.DataFrame(self.training_data)
        return {
            'episodes': df['episode'].values,
            'rewards': df.get('episode_reward', []).values,
            'losses': df.get('policy_loss', []).values,
            'vae_losses': df.get('vae_loss', []).values
        }