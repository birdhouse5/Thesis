# mlflow_integration.py - Combined MLflow setup and comprehensive logging

import os
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import json
import tempfile
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class MLflowIntegration:
    """Combined MLflow setup and comprehensive logging for portfolio optimization experiments."""
    
    def __init__(self, run_name: str = None, config: Dict[str, Any] = None):
        self.run_name = run_name
        self.config = config or {}
        self.episode_count = 0
        self.backend_type = None
        
    def setup_mlflow(self):
        """
        Setup MLflow with MinIO/S3 backend, fallback to local if needed.
        Returns: backend type ('remote' or 'local')
        """
        # Check required environment variables for remote setup
        required_vars = [
            'MLFLOW_TRACKING_URI',
            'AWS_ACCESS_KEY_ID', 
            'AWS_SECRET_ACCESS_KEY',
            'MLFLOW_S3_ENDPOINT_URL'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.warning(f"Missing environment variables for MinIO setup: {missing_vars}")
            logger.info("Falling back to local MLflow tracking")
            return self._setup_local_mlflow()
        
        # Try remote setup
        try:
            return self._setup_remote_mlflow()
        except Exception as e:
            logger.error(f"Remote MLflow setup failed: {e}")
            logger.info("Falling back to local MLflow tracking")
            return self._setup_local_mlflow()
    
    def _setup_remote_mlflow(self):
        """Setup remote MLflow with MinIO."""
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set TLS ignore for MinIO (assuming HTTP not HTTPS)
        os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'
        
        logger.info(f"MLflow configured with remote backend:")
        logger.info(f"  Tracking URI: {tracking_uri}")
        logger.info(f"  S3 Endpoint: {os.getenv('MLFLOW_S3_ENDPOINT_URL')}")
        
        # Test connection
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        logger.info(f"✅ MLflow remote connected. Found {len(experiments)} experiments.")
        
        self.backend_type = 'remote'
        return 'remote'
    
    def _setup_local_mlflow(self):
        """Setup local MLflow fallback."""
        mlflow.set_tracking_uri("file:./mlruns")
        logger.info("✅ MLflow configured with local backend (./mlruns)")
        self.backend_type = 'local'
        return 'local'
    
    def test_connection(self):
        """Test MLflow connection by creating a test run."""
        if not self.backend_type:
            logger.error("MLflow not setup. Call setup_mlflow() first.")
            return False
            
        try:
            with mlflow.start_run(run_name="connection_test"):
                # Test logging a simple metric
                mlflow.log_metric("test_metric", 42.0)
                
                # Test logging a simple artifact
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write("MLflow connection test artifact")
                    test_file = f.name
                
                mlflow.log_artifact(test_file, "test_artifacts")
                os.unlink(test_file)
                
                logger.info(f"✅ MLflow {self.backend_type} backend test successful!")
                return True
                
        except Exception as e:
            logger.error(f"❌ MLflow connection test failed: {e}")
            return False
    
    # ===== COMPREHENSIVE LOGGING METHODS =====
    
    def log_config(self, config: Dict[str, Any] = None):
        """Log all configuration parameters."""
        config_to_log = config or self.config
        
        for key, value in config_to_log.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(key, value)
            else:
                mlflow.log_param(key, str(value))
    
    # def log_training_episode(self, episode: int, metrics: Dict[str, float]):
    #     """Log training episode metrics."""
    #     self.episode_count = episode
        
    #     # Core training metrics
    #     if 'episode_reward' in metrics:
    #         mlflow.log_metric("train_episode_reward", metrics['episode_reward'], step=episode)
    #     if 'policy_loss' in metrics:
    #         mlflow.log_metric("train_policy_loss", metrics['policy_loss'], step=episode)
    #     if 'vae_loss' in metrics:
    #         mlflow.log_metric("train_vae_loss", metrics['vae_loss'], step=episode)
    #     if 'total_steps' in metrics:
    #         mlflow.log_metric("total_training_steps", metrics['total_steps'], step=episode)
            
    #     # Additional training metrics if available
    #     optional_metrics = [
    #         'policy_entropy', 'value_loss', 'grad_norm_policy', 'grad_norm_vae',
    #         'learning_rate_policy', 'learning_rate_vae', 'kl_divergence'
    #     ]
        
    #     for metric in optional_metrics:
    #         if metric in metrics:
    #             mlflow.log_metric(f"train_{metric}", metrics[metric], step=episode)
    
    # def log_portfolio_episode(self, episode: int, portfolio_data: Dict[str, Any]):
    #     """Log detailed portfolio performance for an episode."""
        
    #     # Log everything in results dict
    #     for key, value in portfolio_data.items():
    #         if isinstance(value, (int, float, np.floating)):
    #             mlflow.log_metric(key, float(value), step=episode)

    #     aggregate_keys = [
    #         "episode_avg_reward",
    #         "episode_sum_reward",
    #         "episode_avg_long_exposure",
    #         "episode_avg_short_exposure",
    #         "episode_avg_net_exposure",
    #         "episode_avg_gross_exposure",
    #         "episode_max_active_positions",
    #         "episode_sum_transaction_costs",
    #         "episode_sum_rel_excess_return",
    #     ]

    #     # Portfolio composition
    #     if 'final_weights' in portfolio_data:
    #         weights = portfolio_data['final_weights']
    #         mlflow.log_metric("portfolio_concentration", np.sum(weights**2), step=episode)
    #         mlflow.log_metric("portfolio_active_positions", np.sum(weights > 0.01), step=episode)
    #         mlflow.log_metric("portfolio_long_exposure", np.sum(weights[weights > 0]), step=episode)
    #         mlflow.log_metric("portfolio_short_exposure", abs(np.sum(weights[weights < 0])), step=episode)
    #         mlflow.log_metric("portfolio_net_exposure", np.sum(weights), step=episode)
    #         mlflow.log_metric("portfolio_gross_exposure", np.sum(np.abs(weights)), step=episode)

    #     # Aggregate metrics (safe even if final_weights missing)
    #     for key in aggregate_keys:
    #         if key in portfolio_data:
    #             mlflow.log_metric(key, portfolio_data[key], step=episode)


    #     # Performance metrics
    #     performance_metrics = [
    #         'cumulative_return', 'episode_sharpe', 'episode_volatility', 
    #         'max_drawdown', 'transaction_costs', 'turnover', 'cash_position'
    #     ]
        
    #     for metric in performance_metrics:
    #         if metric in portfolio_data:
    #             mlflow.log_metric(f"portfolio_{metric}", portfolio_data[metric], step=episode)
    
    def log_training_episode(self, episode: int, metrics: Dict[str, Any]):
    """Log all training-related episode metrics."""
    self.episode_count = episode

    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating)):
            mlflow.log_metric(key, float(value), step=episode)


    def log_portfolio_episode(self, episode: int, portfolio_data: Dict[str, Any]):
        """Log all portfolio-related episode metrics, plus derived exposures."""

        # 1. Log every numeric metric
        for key, value in portfolio_data.items():
            if isinstance(value, (int, float, np.floating)):
                mlflow.log_metric(key, float(value), step=episode)

        # 2. Derived metrics from final_weights
        if 'final_weights' in portfolio_data and portfolio_data['final_weights'] is not None:
            weights = np.array(portfolio_data['final_weights'], dtype=float)

            mlflow.log_metric("portfolio_concentration", float(np.sum(weights ** 2)), step=episode)
            mlflow.log_metric("portfolio_active_positions", int(np.sum(weights > 0.01)), step=episode)
            mlflow.log_metric("portfolio_long_exposure", float(np.sum(weights[weights > 0])), step=episode)
            mlflow.log_metric("portfolio_short_exposure", float(abs(np.sum(weights[weights < 0]))), step=episode)
            mlflow.log_metric("portfolio_net_exposure", float(np.sum(weights)), step=episode)
            mlflow.log_metric("portfolio_gross_exposure", float(np.sum(np.abs(weights))), step=episode)




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
    
    def log_final_models(self, policy, encoder=None, experiment_name: str = None):
        """
        Log only final trained models to MinIO (not training checkpoints).
        
        Args:
            policy: Trained policy network
            encoder: Trained encoder network (optional)
            experiment_name: Name for model registration
        """
        exp_name = experiment_name or self.run_name or "portfolio_model"
        
        # Log policy model to MinIO
        mlflow.pytorch.log_model(
            policy, 
            "policy_model",
            registered_model_name=f"{exp_name}_policy"
        )
        logger.info(f"✅ Policy model logged to {self.backend_type} backend")
        
        # Log encoder model if exists
        if encoder is not None:
            mlflow.pytorch.log_model(
                encoder,
                "encoder_model", 
                registered_model_name=f"{exp_name}_encoder"
            )
            logger.info(f"✅ Encoder model logged to {self.backend_type} backend")
    
    def log_essential_artifacts(self, model_dict: Dict[str, torch.nn.Module], 
                              config_dict: Dict[str, Any], experiment_name: str):
        """
        Log essential artifacts: final models + config.
        
        Args:
            model_dict: Dict with 'policy' and optionally 'encoder' PyTorch models
            config_dict: Experiment configuration dictionary  
            experiment_name: Name of the experiment
        """
        # Log final models (only at end of training)
        policy = model_dict.get('policy')
        encoder = model_dict.get('encoder')
        
        self.log_final_models(policy, encoder, experiment_name)
        
        # Log config as JSON artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f, indent=2, default=str)
            config_file = f.name
        
        mlflow.log_artifact(config_file, "config")
        os.unlink(config_file)  # Clean up temp file
        logger.info(f"✅ Configuration logged to {self.backend_type} backend")
    
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
        
        if self.config.get('max_episodes'):
            completion_rate = episodes_completed / self.config['max_episodes']
            mlflow.log_metric("completion_rate", completion_rate)
        
        if error_msg:
            mlflow.log_param("error_message", error_msg[:1000])  # Truncate long error messages

    def log_system_info(self, initial_memory: float, gpu: bool = True):
        import psutil, torch
        mlflow.log_param("system_cpu_count", psutil.cpu_count())
        mlflow.log_param("system_memory_gb", psutil.virtual_memory().total / 1024**3)
        mlflow.log_param("initial_memory_mb", initial_memory)

        if gpu and torch.cuda.is_available():
            mlflow.log_param("gpu_name", torch.cuda.get_device_name())
            mlflow.log_param("gpu_memory_gb", torch.cuda.get_device_properties(0).total_memory / 1024**3)

    def log_final_system_metrics(self, final_memory: float, training_time: float, initial_memory: float):
        mlflow.log_metric("memory_peak_mb", final_memory)
        mlflow.log_metric("memory_increase_mb", final_memory - initial_memory)
        mlflow.log_metric("wall_time_minutes", training_time / 60)

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


# ===== CONVENIENCE FUNCTIONS =====

def setup_mlflow() -> str:
    """
    Simple MLflow setup function for backward compatibility.
    Returns: backend type ('remote' or 'local')
    """
    integrator = MLflowIntegration()
    return integrator.setup_mlflow()

def test_mlflow_connection() -> bool:
    """Test MLflow connection."""
    integrator = MLflowIntegration()
    backend = integrator.setup_mlflow()
    return integrator.test_connection()

def log_essential_artifacts(model_dict: Dict[str, torch.nn.Module], 
                          config_dict: Dict[str, Any], experiment_name: str):
    """Convenience function for logging essential artifacts."""
    integrator = MLflowIntegration()
    integrator.log_essential_artifacts(model_dict, config_dict, experiment_name)

def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


# ===== TESTING =====

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing MLflow integration...")
    success = test_mlflow_connection()
    
    if success:
        print("🎉 MLflow integration working! Ready for experiments.")
    else:
        print("⚠️  Connection test failed. Check your environment variables.")