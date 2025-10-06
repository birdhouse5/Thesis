# csv_logger.py
import pandas as pd
import torch
import json
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

class CSVLogger:
    """CSV-based logger to replace MLflow for simple local experiment tracking."""
    
    def __init__(self, run_name: str, config: Dict[str, Any]):
        self.run_name = run_name
        self.config = config
        self.output_dir = Path("experiment_logs")
        self.output_dir.mkdir(exist_ok=True)
        
        seed = config.get('seed', 0)

        encoder = config.get('encoder', 'unknown')
        asset_class = config.get('asset_class', 'unknown')
        self.output_dir = Path(self.run_name) / encoder / asset_class / "experiment_logs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.csv_path = self.output_dir / f"seed_{seed}_metrics.csv"
        self.config_path = self.output_dir / f"{run_name}_config.json"
        
        # State tracking
        self.headers_written = self.csv_path.exists()
        self.episode_buffer = {}  # Buffer metrics within same episode
        
        # Save config once at initialization
        self._save_config()
        logger.info(f"CSV Logger initialized: {self.csv_path}")
    
    def _save_config(self):
        """Save configuration to JSON file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2, default=self._json_serializer)
            logger.info(f"Config saved: {self.config_path}")
        except Exception as e:
            logger.warning(f"Failed to save config: {e}")
    
    def _json_serializer(self, obj):
        """Handle non-serializable objects for JSON."""
        if isinstance(obj, (np.integer, np.floating, np.ndarray)):
            return obj.item() if hasattr(obj, 'item') else str(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return str(obj)
    
    def _flatten_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dictionaries with dot notation."""
        flattened = {}
        
        for key, value in metrics.items():
            new_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively flatten nested dicts
                flattened.update(self._flatten_metrics(value, new_key))
            elif isinstance(value, (list, tuple)):
                if key == "final_weights" and len(value) <= 50:  # Keep small arrays
                    # Convert to string representation for storage
                    flattened[new_key] = str(value)
                elif key in ["daily_returns", "daily_capital", "daily_weights"]:
                    # Skip large arrays to keep CSV manageable
                    flattened[f"{new_key}_length"] = len(value)
                else:
                    # For other lists, just store length
                    flattened[f"{new_key}_length"] = len(value)
            elif isinstance(value, (int, float, bool, str, type(None))):
                flattened[new_key] = value
            elif isinstance(value, (np.integer, np.floating)):
                flattened[new_key] = float(value)
            else:
                # Convert other types to string
                flattened[new_key] = str(value)
        
        return flattened
    
    def _write_row(self, episode: int, metrics: Dict[str, Any], prefix: str = ""):
        """Write a single row to CSV immediately."""
        try:
            # Flatten the metrics
            flattened = self._flatten_metrics(metrics, prefix)
            
            # Merge with any buffered metrics for this episode
            if episode in self.episode_buffer:
                self.episode_buffer[episode].update(flattened)
                row_data = {
                    'experiment_name': self.run_name,
                    'seed': self.config.get('seed', 0),
                    'asset_class': self.config.get('asset_class', 'unknown'),
                    'encoder': self.config.get('encoder', 'unknown'),
                    'episode': episode, 
                    **self.episode_buffer[episode]
                }
                del self.episode_buffer[episode]  # Clear buffer after writing
            else:
                row_data = {
                    'experiment_name': self.run_name,
                    'seed': self.config.get('seed', 0),
                    'asset_class': self.config.get('asset_class', 'unknown'),
                    'encoder': self.config.get('encoder', 'unknown'),
                    'episode': episode, 
                    **flattened
                }
            
            # Convert to DataFrame for easy CSV handling
            df_row = pd.DataFrame([row_data])
            
            # Always append mode, write headers only if file doesn't exist
            mode = 'a'
            header = not self.headers_written
            
            df_row.to_csv(self.csv_path, index=False, mode=mode, header=header)
            
            if not self.headers_written:
                self.headers_written = True
                logger.info(f"CSV headers written: {len(row_data)} columns")
            
        except Exception as e:
            logger.error(f"Failed to write CSV row for episode {episode}: {e}")
    
    def _buffer_metrics(self, episode: int, metrics: Dict[str, Any], prefix: str = ""):
        """Buffer metrics for an episode (useful when multiple calls per episode)."""
        flattened = self._flatten_metrics(metrics, prefix)
        
        if episode not in self.episode_buffer:
            self.episode_buffer[episode] = {}
        
        self.episode_buffer[episode].update(flattened)
    
    # === Main Logging Interface (matches MLflowIntegration) ===
    
    def log_training_episode(self, episode: int, metrics: Dict[str, Any]):
        """Log training metrics for an episode."""
        self._write_row(episode, metrics, prefix="")
    
    def log_portfolio_episode(self, episode: int, portfolio_data: Dict[str, Any]):
        """Log portfolio-specific metrics."""
        # Since this is called right after training episode, buffer it
        self._buffer_metrics(episode, portfolio_data, prefix="portfolio")
    
    def log_validation_results(self, episode: int, val_results: Dict[str, float]):
        """Log validation results."""
        self._write_row(episode, val_results, prefix="val")
    
    def log_backtest_results(self, backtest_results: Dict[str, Any]):
        """Log final backtest results to separate file."""
        try:
            backtest_path = self.output_dir / f"{self.run_name}_backtest.json"
            with open(backtest_path, 'w') as f:
                json.dump(backtest_results, f, indent=2, default=self._json_serializer)
            logger.info(f"Backtest results saved: {backtest_path}")
        except Exception as e:
            logger.warning(f"Failed to save backtest results: {e}")
    
    # === Compatibility Methods (MLflow interface stubs) ===
    
    def setup_mlflow(self):
        """Compatibility method - returns 'csv' as backend type."""
        return "csv"
    
    def log_config(self, config: Dict[str, Any] = None):
        """Config already saved in __init__, this is a no-op for compatibility."""
        pass
    
    def log_essential_artifacts(self, model_dict: Dict, config_dict: Dict, experiment_name: str):
        """Save model state dictionaries to files."""
        try:
            artifacts_dir = self.output_dir / f"{self.run_name}_artifacts"
            artifacts_dir.mkdir(exist_ok=True)
            
            # Save model state dicts
            for name, model in model_dict.items():
                if model is not None:
                    model_path = artifacts_dir / f"{name}_state_dict.pt"
                    torch.save(model.state_dict(), model_path)
                    logger.info(f"Model saved: {model_path}")
            
            # Save config again in artifacts dir
            config_path = artifacts_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=self._json_serializer)
                
        except Exception as e:
            logger.warning(f"Failed to save artifacts: {e}")
    
    def log_final_summary(self, success: bool, episodes_completed: int, error_msg: str = None):
        """Log final experiment summary."""
        summary = {
            "success": success,
            "episodes_completed": episodes_completed,
            "experiment_name": self.run_name,
            "error_message": error_msg
        }
        
        try:
            summary_path = self.output_dir / f"{self.run_name}_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Final summary saved: {summary_path}")
        except Exception as e:
            logger.warning(f"Failed to save summary: {e}")
    
    def log_system_info(self, initial_memory: float, gpu: bool = True):
        """Log system information to summary."""
        pass  # Could extend this if needed
    
    def log_final_system_metrics(self, final_memory: float, training_time: float, initial_memory: float):
        """Log final system metrics to summary."""
        pass  # Could extend this if needed

# === Helper Functions ===

def create_csv_logger(run_name: str, config: Dict[str, Any]) -> CSVLogger:
    """Factory function to create CSV logger."""
    return CSVLogger(run_name, config)


class TrainingCSVLogger:
    """Training-specific CSV logger with fixed schema."""
    
    def __init__(self, experiment_name: str, seed: int, asset_class: str, encoder: str):
        self.experiment_name = experiment_name
        self.seed = seed
        self.asset_class = asset_class
        self.encoder = encoder
        
        self.output_dir = Path(experiment_name) / encoder / asset_class / "experiment_logs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / f"{experiment_name}_training.csv"
        
        # Write headers if file doesn't exist
        if not self.csv_path.exists():
            self._write_headers()
    
    def _write_headers(self):
        # headers = [
        #     'experiment_name', 'seed', 'asset_class', 'encoder', 'episode',
        #     'policy_loss', 'value_loss', 'entropy', 'vae_loss',
        #     'vae_recon_obs', 'vae_recon_reward', 'vae_kl', 'vae_total', 
        #     'vae_context_len', 'vae_latent_mu_mean', 'vae_latent_logvar_mean',
        #     'hmm_converged', 'hmm_log_likelihood', 
        #     'hmm_regime_0_prob', 'hmm_regime_1_prob', 'hmm_regime_2_prob', 'hmm_regime_3_prob',
        #     'episode_sum_reward', 'episode_final_capital', 'episode_total_return', 
        #     'steps_per_episode', 'episode_count'
        # ]
        headers = [
            'experiment_name', 'seed', 'asset_class', 'encoder', 'cumulative_episodes',
            'policy_loss', 'value_loss', 'entropy', 'advantages_mean', 'advantages_std', 
            'advantages_min', 'advantages_max', 'vae_loss',
            'vae_recon_obs', 'vae_recon_reward', 'vae_kl', 'vae_total', 
            'vae_context_len', 'vae_latent_mu_mean', 'vae_latent_logvar_mean',
            'hmm_converged', 'hmm_log_likelihood', 
            'hmm_regime_0_prob', 'hmm_regime_1_prob', 'hmm_regime_2_prob', 'hmm_regime_3_prob',
            # Task-level metrics (what you actually have)
            'task_total_reward', 'task_avg_reward_per_episode',
            'task_final_capital', 'task_cumulative_return',
            'total_steps', 'episodes_per_task', 'task_count'
        ]

        with open(self.csv_path, 'w') as f:
            f.write(','.join(headers) + '\n')
    

    def log_task(self, task: int, metrics: Dict[str, Any]):
        """Log one training episode or task summary."""
        logger = logging.getLogger(__name__)
        logger.info(f"=== TrainingCSVLogger.log_episode DEBUG ===")
        logger.info(f"  Task: {task}")
        logger.info(f"  Received metrics keys: {list(metrics.keys())}")

        # Extract VAE and HMM metrics
        vae_metrics = {k[4:]: v for k, v in metrics.items() if k.startswith('vae_')}
        hmm_metrics = {k[4:]: v for k, v in metrics.items() if k.startswith('hmm_')}

        # Fallbacks for expected metrics
        row = [
            self.experiment_name, self.seed, self.asset_class, self.encoder, task,
            metrics.get('policy_loss', 0.0),
            metrics.get('value_loss', 0.0),
            metrics.get('entropy', 0.0),
            metrics.get('advantages_mean', 0.0),
            metrics.get('advantages_std', 0.0),
            metrics.get('advantages_min', 0.0),
            metrics.get('advantages_max', 0.0),
            metrics.get('vae_loss', 0.0),
            vae_metrics.get('recon_obs', 0.0),
            vae_metrics.get('recon_reward', 0.0),
            vae_metrics.get('kl', 0.0),
            vae_metrics.get('total', 0.0),
            vae_metrics.get('context_len', 0),
            vae_metrics.get('latent_mu_mean', 0.0),
            vae_metrics.get('latent_logvar_mean', 0.0),
            hmm_metrics.get('converged', 0),
            hmm_metrics.get('log_likelihood', 0.0),
            hmm_metrics.get('regime_0_prob', 0.0),
            hmm_metrics.get('regime_1_prob', 0.0),
            hmm_metrics.get('regime_2_prob', 0.0),
            hmm_metrics.get('regime_3_prob', 0.0),

            # ✅ Now include task-level metrics
            metrics.get('task_total_reward', 0.0),
            metrics.get('task_avg_reward_per_episode', 0.0),
            metrics.get('task_final_capital', 0.0),
            metrics.get('task_cumulative_return', 0.0),
            metrics.get('total_steps', 0),
            metrics.get('episodes_per_task', 0),
            metrics.get('task_count', 0),
        ]

        with open(self.csv_path, 'a') as f:
            f.write(','.join(map(str, row)) + '\n')

    # def log_episode(self, episode: int, metrics: Dict[str, Any]):
    #     """Log one training episode."""

    #     logger = logging.getLogger(__name__)
    #     logger.info(f"=== TrainingCSVLogger.log_episode DEBUG ===")
    #     logger.info(f"  Episode: {episode}")
    #     logger.info(f"  Received metrics keys: {list(metrics.keys())}")
    #     logger.info(f"  Received metrics values: {metrics}")

    #     # Extract VAE metrics if present
    #     vae_metrics = {k[4:]: v for k, v in metrics.items() if k.startswith('vae_')}
    #     logger.info(f"  Extracted VAE metrics: {vae_metrics}")
        
    #     # Extract HMM metrics if present (would need to be passed from trainer)
    #     hmm_metrics = {k[4:]: v for k, v in metrics.items() if k.startswith('hmm_')}
        
    #     expected_fields = [
    #         'policy_loss', 'value_loss', 'entropy', 'vae_loss',
    #         'episode_sum_reward', 'episode_final_capital', 'episode_total_return',
    #         'steps_per_episode', 'episode_count'
    #     ]
    #     missing_fields = [f for f in expected_fields if f not in metrics]
    #     if missing_fields:
    #         logger.warning(f"  MISSING EXPECTED FIELDS: {missing_fields}")

    #     row = [
    #         self.experiment_name, self.seed, self.asset_class, self.encoder, episode,
    #         metrics.get('policy_loss', 0.0),
    #         metrics.get('value_loss', 0.0), 
    #         metrics.get('entropy', 0.0),
    #         metrics.get('vae_loss', 0.0),
    #         vae_metrics.get('recon_obs', 0.0),
    #         vae_metrics.get('recon_reward', 0.0),
    #         vae_metrics.get('kl', 0.0),
    #         vae_metrics.get('total', 0.0),
    #         vae_metrics.get('context_len', 0),
    #         vae_metrics.get('latent_mu_mean', 0.0),
    #         vae_metrics.get('latent_logvar_mean', 0.0),
    #         hmm_metrics.get('converged', 0),
    #         hmm_metrics.get('log_likelihood', 0.0),
    #         hmm_metrics.get('regime_0_prob', 0.0),
    #         hmm_metrics.get('regime_1_prob', 0.0), 
    #         hmm_metrics.get('regime_2_prob', 0.0),
    #         hmm_metrics.get('regime_3_prob', 0.0),
    #         metrics.get('episode_sum_reward', 0.0),
    #         metrics.get('episode_final_capital', 0.0),
    #         metrics.get('episode_total_return', 0.0),
    #         metrics.get('steps_per_episode', 0),
    #         metrics.get('episode_count', 0)
    #     ]
        
    #     with open(self.csv_path, 'a') as f:
    #         f.write(','.join(map(str, row)) + '\n')

class ValidationCSVLogger:
    """Validation-specific CSV logger with fixed schema."""
    
    def __init__(self, experiment_name: str, seed: int, asset_class: str, encoder: str):
        self.experiment_name = experiment_name
        self.seed = seed
        self.asset_class = asset_class
        self.encoder = encoder
        
        self.output_dir = Path(experiment_name) / encoder / asset_class / "experiment_logs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / f"{experiment_name}_validation.csv"
        
        # Write headers if file doesn't exist
        if not self.csv_path.exists():
            self._write_headers()
    
    def _write_headers(self):
        headers = [
            'experiment_name', 'seed', 'asset_class', 'encoder', 'episode',
            'avg_reward', 'std_reward', 'avg_return', 'std_return', 
            'avg_volatility', 'avg_episode_sharpe', 'max_return', 'min_return', 
            'num_episodes'
        ]
        
        with open(self.csv_path, 'w') as f:
            f.write(','.join(headers) + '\n')
    
    def log_validation(self, episode: int, val_results: Dict[str, Any]):
        """Log validation results."""
        # Remove "validation: " prefix from keys
        clean_results = {}
        for k, v in val_results.items():
            clean_key = k.replace('validation: ', '') if k.startswith('validation: ') else k
            clean_results[clean_key] = v
        
        row = [
            self.experiment_name, self.seed, self.asset_class, self.encoder, episode,
            clean_results.get('avg_reward', 0.0),
            clean_results.get('std_reward', 0.0),
            clean_results.get('avg_return', 0.0),
            clean_results.get('std_return', 0.0),
            clean_results.get('avg_volatility', 0.0),
            clean_results.get('avg_episode_sharpe', 0.0),
            clean_results.get('max_return', 0.0),
            clean_results.get('min_return', 0.0),
            clean_results.get('num_episodes', 0)
        ]
        
        with open(self.csv_path, 'a') as f:
            f.write(','.join(map(str, row)) + '\n')


class BacktestCSVLogger:
    """Backtest-specific CSV logger - one row per time step."""
    
    def __init__(self, experiment_name: str, seed: int, asset_class: str, encoder: str, num_assets: int, latent_dim: int):
        self.latent_dim = latent_dim
        self.experiment_name = experiment_name
        self.seed = seed
        self.asset_class = asset_class
        self.encoder = encoder
        self.num_assets = num_assets
        
        self.output_dir = Path(experiment_name) / encoder / asset_class / "experiment_logs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / f"{experiment_name}_backtest.csv"
        
        # Write headers if file doesn't exist
        if not self.csv_path.exists():
            self._write_headers()
    
    def _write_headers(self):
        headers = [
            'experiment_name', 'seed', 'asset_class', 'encoder', 'step',
            'capital', 'log_return', 'excess_return', 'reward', 
            'long_exposure', 'short_exposure', 'cash_position', 'net_exposure', 
            'gross_exposure', 'turnover', 'transaction_cost'
        ]

        # Add weight columns: weight_0, weight_1, ..., weight_N
        for i in range(self.num_assets):
            headers.append(f'weight_{i}')

        # Add latent columns
        for i in range(self.latent_dim):
            headers.append(f'latent_{i}')
        
        with open(self.csv_path, 'w') as f:
            f.write(','.join(headers) + '\n')
    
    def log_step(self, step: int, capital: float, log_return: float, excess_return: float, 
                 reward: float, weights: np.ndarray, long_exposure: float, short_exposure: float,
                 cash_position: float, net_exposure: float, gross_exposure: float, turnover: float, 
                 transaction_cost: float, latent: list):
        """Log one backtest time step."""
        
        row = [
            self.experiment_name, self.seed, self.asset_class, self.encoder, step,
            capital, log_return, excess_return, reward,
            long_exposure, short_exposure, cash_position, net_exposure, gross_exposure,
            turnover, transaction_cost
        ]
        
        # Add individual weights
        for weight in weights:
            row.append(weight)
        
        # Add latent values
        for i in range(self.latent_dim):
            row.append(latent[i])
        
        with open(self.csv_path, 'a') as f:
            f.write(','.join(map(str, row)) + '\n')