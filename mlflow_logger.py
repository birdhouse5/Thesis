# mlflow_logger.py - Simplified MLflow integration (no MinIO/S3)

import os
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import json
import tempfile
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class MLflowIntegration:
    """Simplified MLflow logging for portfolio optimization experiments."""

    def __init__(self, run_name: str = None, config: Dict[str, Any] = None):
        self.run_name = run_name
        self.config = config or {}
        self.episode_count = 0
        self.backend_type = None

    def setup_mlflow(self):
        if os.getenv("DISABLE_MLFLOW", "false").lower() == "true":
            logger.info("🚫 MLflow disabled by DISABLE_MLFLOW")
            self.backend_type = "disabled"

            # Monkey-patch mlflow to no-ops so callers don’t crash
            mlflow.start_run = contextlib.nullcontext
            mlflow.active_run = lambda: None
            mlflow.log_param = lambda *a, **k: None
            mlflow.log_metric = lambda *a, **k: None
            mlflow.set_tracking_uri = lambda *a, **k: None
            mlflow.pytorch.log_model = lambda *a, **k: None
            return "disabled"

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"✅ MLflow configured with remote tracking URI: {tracking_uri}")
            self.backend_type = "remote"
            return "remote"
        else:
            mlflow.set_tracking_uri("file:./mlruns")
            logger.info("✅ MLflow configured with local backend (./mlruns)")
            self.backend_type = "local"
            return "local"

    def test_connection(self):
        """Test MLflow connection by creating a test run."""
        if not self.backend_type:
            logger.error("MLflow not setup. Call setup_mlflow() first.")
            return False

        try:
            with mlflow.start_run(run_name="connection_test"):
                mlflow.log_metric("test_metric", 42.0)
                with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                    f.write("MLflow connection test artifact")
                    test_file = f.name
                mlflow.log_artifact(test_file, "test_artifacts")
                os.unlink(test_file)

            logger.info(f"✅ MLflow {self.backend_type} backend test successful!")
            return True
        except Exception as e:
            logger.error(f"❌ MLflow connection test failed: {e}")
            return False

    # ===== Logging methods =====

    def log_config(self, config: Dict[str, Any] = None):
        """Log all configuration parameters."""
        if self.backend_type == "disabled": return
        config_to_log = config or self.config
        for key, value in config_to_log.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(key, value)
            else:
                mlflow.log_param(key, str(value))

    def log_training_episode(self, episode: int, metrics: Dict[str, Any]):
        """Log training metrics."""
        if self.backend_type == "disabled": return
        self.episode_count = episode
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.floating)):
                mlflow.log_metric(key, float(value), step=episode)

    def log_portfolio_episode(self, episode: int, portfolio_data: Dict[str, Any]):
        """Log portfolio metrics + derived exposures."""
        if self.backend_type == "disabled": return
        for key, value in portfolio_data.items():
            if isinstance(value, (int, float, np.floating)):
                mlflow.log_metric(key, float(value), step=episode)

        if "final_weights" in portfolio_data and portfolio_data["final_weights"] is not None:
            weights = np.array(portfolio_data["final_weights"], dtype=float)
            mlflow.log_metric("portfolio_concentration", float(np.sum(weights ** 2)), step=episode)
            mlflow.log_metric("portfolio_active_positions", int(np.sum(weights > 0.01)), step=episode)
            mlflow.log_metric("portfolio_long_exposure", float(np.sum(weights[weights > 0])), step=episode)
            mlflow.log_metric("portfolio_short_exposure", float(abs(np.sum(weights[weights < 0]))), step=episode)
            mlflow.log_metric("portfolio_net_exposure", float(np.sum(weights)), step=episode)
            mlflow.log_metric("portfolio_gross_exposure", float(np.sum(np.abs(weights))), step=episode)

    def log_validation_results(self, episode: int, val_results: Dict[str, float]):
        if self.backend_type == "disabled": return
        core_metrics = ["avg_reward", "std_reward", "avg_return", "std_return", "avg_volatility", "avg_sharpe"]
        for metric in core_metrics:
            if metric in val_results:
                mlflow.log_metric(f"val_{metric}", val_results[metric], step=episode)

    def log_backtest_results(self, backtest_results: Dict[str, Any]):
        if self.backend_type == "disabled": return
        core_metrics = [
            "total_return", "annual_return", "annual_volatility", "sharpe_ratio",
            "sortino_ratio", "calmar_ratio", "max_drawdown", "win_rate", "var_95"
        ]
        for metric in core_metrics:
            if metric in backtest_results:
                mlflow.log_metric(f"backtest_{metric}", backtest_results[metric])

        if "daily_returns" in backtest_results and isinstance(backtest_results["daily_returns"], list):
            returns_file = "daily_returns.json"
            with open(returns_file, "w") as f:
                json.dump(backtest_results["daily_returns"], f)
            mlflow.log_artifact(returns_file)

    def log_final_models(self, policy, encoder=None, experiment_name: str = None):
        if self.backend_type == "disabled": return
        """Log final trained models."""
        exp_name = experiment_name or self.run_name or "portfolio_model"
        mlflow.pytorch.log_model(policy, "policy_model", registered_model_name=f"{exp_name}_policy")
        logger.info(f"✅ Policy model logged to {self.backend_type} backend")
        if encoder is not None:
            mlflow.pytorch.log_model(encoder, "encoder_model", registered_model_name=f"{exp_name}_encoder")
            logger.info(f"✅ Encoder model logged to {self.backend_type} backend")

    def log_essential_artifacts(self, model_dict: Dict[str, torch.nn.Module], config_dict: Dict[str, Any], experiment_name: str):
        if self.backend_type == "disabled": return
        self.log_final_models(model_dict.get("policy"), model_dict.get("encoder"), experiment_name)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_dict, f, indent=2, default=str)
            config_file = f.name
        mlflow.log_artifact(config_file, "config")
        os.unlink(config_file)
        logger.info(f"✅ Configuration logged to {self.backend_type} backend")

    def log_final_summary(self, success: bool, episodes_completed: int, error_msg: str = None):
        if self.backend_type == "disabled": return
        mlflow.log_metric("experiment_success", 1 if success else 0)
        mlflow.log_metric("episodes_completed", episodes_completed)
        if self.config.get("max_episodes"):
            completion_rate = episodes_completed / self.config["max_episodes"]
            mlflow.log_metric("completion_rate", completion_rate)
        if error_msg:
            mlflow.log_param("error_message", error_msg[:1000])

    
    def log_system_info(self, initial_memory: float, gpu: bool = True):
        if self.backend_type == "disabled": return
        import psutil, torch
        mlflow.log_param("system_cpu_count", psutil.cpu_count())
        mlflow.log_param("system_memory_gb", psutil.virtual_memory().total / 1024**3)
        mlflow.log_param("initial_memory_mb", initial_memory)

        if gpu and torch.cuda.is_available():
            mlflow.log_param("gpu_name", torch.cuda.get_device_name())
            mlflow.log_param("gpu_memory_gb", torch.cuda.get_device_properties(0).total_memory / 1024**3)

    def log_final_system_metrics(self, final_memory: float, training_time: float, initial_memory: float):
        if self.backend_type == "disabled": return
        mlflow.log_metric("memory_peak_mb", final_memory)
        mlflow.log_metric("memory_increase_mb", final_memory - initial_memory)
        mlflow.log_metric("wall_time_minutes", training_time / 60)

# ===== Convenience functions =====

def setup_mlflow() -> str:
    integrator = MLflowIntegration()
    return integrator.setup_mlflow()

def test_mlflow_connection() -> bool:
    integrator = MLflowIntegration()
    backend = integrator.setup_mlflow()
    return integrator.test_connection()
