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
        
        # File paths
        self.csv_path = self.output_dir / f"{run_name}_metrics.csv"
        self.config_path = self.output_dir / f"{run_name}_config.json"
        
        # State tracking
        self.headers_written = False
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
                row_data = {'episode': episode, **self.episode_buffer[episode]}
                del self.episode_buffer[episode]  # Clear buffer after writing
            else:
                row_data = {'episode': episode, **flattened}
                row_data['seed'] = self.config.get('seed', 0)
            
            # Convert to DataFrame for easy CSV handling
            df_row = pd.DataFrame([row_data])
            
            # Write to CSV (header only on first write)
            mode = 'w' if not self.headers_written else 'a'
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