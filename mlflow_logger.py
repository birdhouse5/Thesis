import mlflow
import mlflow.pytorch
import numpy as np
import torch
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ComprehensiveMLflowLogger:
    """Comprehensive logging for portfolio optimization experiments."""
    
    def __init__(self, run_name: str, config: Dict[str, Any]):
        self.run_name = run_name
        self.config = config
        self.episode_count = 0
        
    def log_config(self):
        """Log all configuration parameters."""
        for key, value in self.config.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(key, value)
            else:
                mlflow.log_param(key, str(value))
    
    def log_training_episode(self, episode: int, metrics: Dict[str, float]):
        """Log training episode metrics."""
        self.episode_count = episode
        
        # Core training metrics
        if 'episode_reward' in metrics:
            mlflow.log_metric("train_episode_reward", metrics['episode_reward'], step=episode)
        if 'policy_loss' in metrics:
            mlflow.log_metric("train_policy_loss", metrics['policy_loss'], step=episode)
        if 'vae_loss' in metrics:
            mlflow.log_metric("train_vae_loss", metrics['vae_loss'], step=episode)
        if 'total_steps' in metrics:
            mlflow.log_metric("total_training_steps", metrics['total_steps'], step=episode)
            
        # Additional training metrics if available
        optional_metrics = [
            'policy_entropy', 'value_loss', 'grad_norm_policy', 'grad_norm_vae',
            'learning_rate_policy', 'learning_rate_vae', 'kl_divergence'
        ]
        
        for metric in optional_metrics:
            if metric in metrics:
                mlflow.log_metric(f"train_{metric}", metrics[metric], step=episode)
    
    def log_portfolio_episode(self, episode: int, portfolio_data: Dict[str, Any]):
        """Log detailed portfolio performance for an episode."""
        
        # Portfolio composition
        if 'final_weights' in portfolio_data:
            weights = portfolio_data['final_weights']
            mlflow.log_metric("portfolio_concentration", np.sum(weights**2), step=episode)  # HHI
            mlflow.log_metric("portfolio_active_positions", np.sum(weights > 0.01), step=episode)
            mlflow.log_metric("portfolio_long_exposure", np.sum(weights[weights > 0]), step=episode)
            mlflow.log_metric("portfolio_short_exposure", abs(np.sum(weights[weights < 0])), step=episode)
        
        # Performance metrics
        performance_metrics = [
            'cumulative_return', 'episode_sharpe', 'episode_volatility', 
            'max_drawdown', 'transaction_costs', 'turnover', 'cash_position'
        ]
        
        for metric in performance_metrics:
            if metric in portfolio_data:
                mlflow.log_metric(f"portfolio_{metric}", portfolio_data[metric], step=episode)
    
    def log_validation_results(self, episode: int, val_results: Dict[str, float]):
        """Log validation evaluation results."""
        
        # Standard validation metrics
        core_metrics = ['avg_reward', 'std_reward', 'avg_return', 'std_return', 'avg_volatility', 'avg_sharpe']
        
        for metric in core_metrics:
            if metric in val_results:
                mlflow.log_metric(f"val_{metric}", val_results[metric], step=episode)
        
        # Extended validation metrics
        extended_metrics = [
            'max_return', 'min_return', 'win_rate', 'profit_factor',
            'avg_drawdown', 'recovery_time', 'consistency_ratio'
        ]
        
        for metric in extended_metrics:
            if metric in val_results:
                mlflow.log_metric(f"val_{metric}", val_results[metric], step=episode)
    
    def log_backtest_results(self, backtest_results: Dict[str, Any]):
        """Log comprehensive backtest results."""
        
        # Core performance metrics
        core_metrics = [
            'total_return', 'annual_return', 'annual_volatility', 'sharpe_ratio',
            'sortino_ratio', 'calmar_ratio', 'max_drawdown', 'win_rate', 'var_95'
        ]
        
        for metric in core_metrics:
            if metric in backtest_results:
                mlflow.log_metric(f"backtest_{metric}", backtest_results[metric])
        
        # Risk metrics
        risk_metrics = [
            'downside_deviation', 'upside_deviation', 'skewness', 'kurtosis',
            'tail_ratio', 'conditional_var', 'maximum_loss'
        ]
        
        for metric in risk_metrics:
            if metric in backtest_results:
                mlflow.log_metric(f"backtest_{metric}", backtest_results[metric])
        
        # Portfolio metrics
        portfolio_metrics = [
            'avg_concentration', 'avg_turnover', 'total_transaction_costs',
            'avg_leverage', 'tracking_error', 'information_ratio'
        ]
        
        for metric in portfolio_metrics:
            if metric in backtest_results:
                mlflow.log_metric(f"backtest_{metric}", backtest_results[metric])
        
        # Log detailed time series (as artifacts)
        if 'daily_returns' in backtest_results and isinstance(backtest_results['daily_returns'], list):
            returns_array = np.array(backtest_results['daily_returns'])
            
            # Save returns as artifact
            returns_file = "daily_returns.json"
            with open(returns_file, 'w') as f:
                json.dump(backtest_results['daily_returns'], f)
            mlflow.log_artifact(returns_file)
            
            # Log distribution statistics
            mlflow.log_metric("backtest_returns_mean", np.mean(returns_array))
            mlflow.log_metric("backtest_returns_std", np.std(returns_array))
            mlflow.log_metric("backtest_returns_skew", self._safe_skew(returns_array))
            mlflow.log_metric("backtest_returns_kurt", self._safe_kurtosis(returns_array))
        
        # Log portfolio weights evolution
        if 'daily_weights' in backtest_results and isinstance(backtest_results['daily_weights'], list):
            weights_file = "daily_weights.json"
            with open(weights_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_weights = []
                for w in backtest_results['daily_weights']:
                    if isinstance(w, np.ndarray):
                        serializable_weights.append(w.tolist())
                    else:
                        serializable_weights.append(w)
                json.dump(serializable_weights, f)
            mlflow.log_artifact(weights_file)
    
    def log_model_artifacts(self, policy, encoder=None, model_dir: str = "models"):
        """Log trained models as MLflow artifacts."""
        
        # Log policy model
        mlflow.pytorch.log_model(
            policy, 
            "policy_model",
            registered_model_name=f"{self.run_name}_policy"
        )
        
        # Log encoder model if exists
        if encoder is not None:
            mlflow.pytorch.log_model(
                encoder,
                "encoder_model", 
                registered_model_name=f"{self.run_name}_encoder"
            )
    
    def log_system_metrics(self, gpu_memory_mb: float, training_time_sec: float):
        """Log system performance metrics."""
        mlflow.log_metric("gpu_memory_peak_mb", gpu_memory_mb)
        mlflow.log_metric("training_time_seconds", training_time_sec)
        mlflow.log_metric("training_time_minutes", training_time_sec / 60)
        mlflow.log_metric("training_time_hours", training_time_sec / 3600)
    
    def log_experiment_metadata(self, experiment_id: int, total_experiments: int):
        """Log experiment batch metadata."""
        mlflow.log_param("experiment_id", experiment_id)
        mlflow.log_param("total_experiments", total_experiments)
        mlflow.log_param("experiment_timestamp", datetime.now().isoformat())
    
    def log_final_summary(self, success: bool, episodes_completed: int, error_msg: str = None):
        """Log final experiment summary."""
        mlflow.log_metric("experiment_success", 1 if success else 0)
        mlflow.log_metric("episodes_completed", episodes_completed)
        mlflow.log_metric("completion_rate", episodes_completed / self.config.get('max_episodes', 1))
        
        if error_msg:
            mlflow.log_param("error_message", error_msg[:1000])  # Truncate long error messages
    
    @staticmethod
    def _safe_skew(data):
        """Compute skewness safely."""
        try:
            from scipy.stats import skew
            return float(skew(data))
        except:
            return 0.0
    
    @staticmethod 
    def _safe_kurtosis(data):
        """Compute kurtosis safely."""
        try:
            from scipy.stats import kurtosis
            return float(kurtosis(data))
        except:
            return 0.0


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0