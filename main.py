import mlflow
import mlflow.pytorch
from pathlib import Path
from datetime import datetime
import torch
import logging
import traceback
import numpy as np
from typing import Dict, Any, Optional
import os
import json
import time
import pandas as pd

# --- import your config system ---
from config import (
    generate_experiment_configs,
    experiment_to_training_config,
    TrainingConfig
)

# --- import your env + models + trainer ---
from environments.data import PortfolioDataset
from environments.env import MetaEnv
from models.policy import PortfolioPolicy
from models.vae import VAE
from models.hmm_encoder import HMMEncoder  # New stub
from algorithms.trainer import PPOTrainer

# Import evaluation functions
from evaluation_backtest import evaluate, run_sequential_backtest

# Set up logging
def setup_debug_logging():
    """Configure logging based on DEBUG environment variable."""
    
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    test_mode = os.getenv("TEST_MODE", "false").lower() == "true"
    
    # Set log level
    if debug_mode:
        log_level = logging.DEBUG
        log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    else:
        log_level = logging.INFO
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler('experiment_debug.log', mode='w')  # File output
        ],
        force=True  # Override existing configuration
    )
    
    # Set specific logger levels for debug mode
    if debug_mode:
        # Enable debug for your modules
        logging.getLogger('experiment_manager').setLevel(logging.DEBUG)
        logging.getLogger('resource_manager').setLevel(logging.DEBUG)
        logging.getLogger('mlflow_logger').setLevel(logging.DEBUG)
        
        # Reduce noise from external libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('mlflow').setLevel(logging.INFO)
        
        logger = logging.getLogger(__name__)
        logger.info("DEBUG MODE ENABLED")
        logger.debug(f"Debug logging configured - output to console and experiment_debug.log")
        
        if test_mode:
            logger.info("TEST MODE ENABLED - running limited experiments")
    
    return debug_mode, test_mode

# Initialize logging and get modes
debug_mode, test_mode = setup_debug_logging()
logger = logging.getLogger(__name__)

# Import additional modules
from experiment_manager import ExperimentManager

def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cleanup_gpu_memory():
    """Clean up GPU memory and cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def ensure_dataset_exists(cfg: TrainingConfig) -> str:
    """Ensure the required dataset exists, creating it if necessary."""
    data_path = Path(cfg.data_path)
    
    if data_path.exists():
        logger.info(f"Dataset exists: {data_path}")
        return str(data_path)
    
    logger.info(f"Creating {cfg.asset_class} dataset...")
    
    PortfolioDataset(
        asset_class=cfg.asset_class,
        data_path=str(data_path),
        force_recreate=True  # ensures new file is created
    )

    return str(data_path)


def prepare_environments(cfg: TrainingConfig):
    """Prepare train/val/test environments from dataset."""
    
    # Create unified dataset - handles both crypto intelligent splitting and SP500
    if cfg.asset_class == "crypto":
        # For crypto, use proportional splitting (more reliable than date-based)
        dataset = PortfolioDataset(
            asset_class=cfg.asset_class,
            data_path=cfg.data_path,
            proportional=True,
            proportions=(0.7, 0.2, 0.1)
        )
        # Update config with computed dates for consistency
        cfg.train_end = dataset.train_end
        cfg.val_end = dataset.val_end
        logger.info(f"Crypto dataset with intelligent splits: train_end={cfg.train_end}, val_end={cfg.val_end}")
    else:
        # SP500 uses explicit date-based approach
        dataset = PortfolioDataset(
            asset_class=cfg.asset_class,
            data_path=cfg.data_path,
            train_end=cfg.train_end,
            val_end=cfg.val_end,
            proportional=False
        )
    
    # Get all splits
    datasets = dataset.get_all_splits()
    
    # Update num_assets from actual data
    cfg.num_assets = datasets['train'].num_assets
    
    # Convert datasets to tensor format for MetaEnv (simplified)
    split_tensors = {}
    for split_name, dataset_split in datasets.items():
        features_list, prices_list = [], []
        num_windows = max(1, (len(dataset_split) - cfg.seq_len) // cfg.seq_len)
        
        for i in range(num_windows):
            start, end = i * cfg.seq_len, (i+1) * cfg.seq_len
            if end <= len(dataset_split):
                # Use new direct tensor method - eliminates intermediate conversions
                window_tensors = dataset_split.get_window_tensor(start, end, device='cpu')
                features_list.append(window_tensors['features'])
                prices_list.append(window_tensors['raw_prices'])
        
        if len(features_list) == 0:
            raise ValueError(f"No complete windows available for {split_name} split")
        
        all_features = torch.stack(features_list)
        all_prices = torch.stack(prices_list)
        
        split_tensors[split_name] = {
            'features': all_features.view(-1, cfg.num_assets, dataset_split.num_features),
            'raw_prices': all_prices.view(-1, cfg.num_assets),
            'feature_columns': dataset_split.feature_cols,
            'num_windows': len(features_list)
        }
    
    # Determine steps_per_year based on asset class
    if cfg.asset_class == "crypto":
        steps_per_year = 35040  # 15-minute intervals: 365 * 24 * 4
    else:
        steps_per_year = 252    # Daily intervals for stocks
    
    # Create environments with DSR parameters from config
    environments = {}
    for split_name, tensor_data in split_tensors.items():
        environments[split_name] = MetaEnv(
            dataset={
                'features': tensor_data['features'],
                'raw_prices': tensor_data['raw_prices']
            },
            feature_columns=tensor_data['feature_columns'],
            seq_len=cfg.seq_len,
            min_horizon=cfg.min_horizon,
            max_horizon=cfg.max_horizon,
            eta=cfg.eta,
            rf_rate=cfg.rf_rate,
            transaction_cost_rate=cfg.transaction_cost_rate,
            steps_per_year=steps_per_year
        )
    
    return environments, split_tensors

def create_models(cfg: TrainingConfig, obs_shape) -> tuple:
    """Create encoder and policy models based on configuration."""
    device = torch.device(cfg.device)
    
    # Create encoder based on type
    encoder = None
    if cfg.encoder == "vae":
        encoder = VAE(
            obs_dim=obs_shape,
            num_assets=cfg.num_assets,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim
        ).to(device)
    elif cfg.encoder == "hmm":
        import mlflow.pytorch
        model_name = f"{cfg.asset_class}_hmm_encoder"
        try:
            encoder = mlflow.pytorch.load_model(f"models:/{model_name}/latest").to(device)
            logger.info(f"✅ Loaded pretrained HMM encoder from MinIO: {model_name}")
        except Exception as e:
            logger.error(f"Required pretrained HMM encoder not found: {e}")
            raise RuntimeError(f"HMM encoder '{model_name}' must be pretrained first. Run pretrain_hmm.py")

    # encoder == "none" -> encoder remains None
    
    # Create policy
    policy = PortfolioPolicy(
        obs_shape=obs_shape,
        latent_dim=cfg.latent_dim,
        num_assets=cfg.num_assets,
        hidden_dim=cfg.hidden_dim
    ).to(device)
    
    return encoder, policy

def run_training(cfg: TrainingConfig) -> Dict[str, Any]:
    """Run complete training pipeline with comprehensive logging."""
    
    try:
        # Setup
        seed_everything(cfg.seed)
        
        # Ensure dataset exists
        cfg.data_path = ensure_dataset_exists(cfg)
        
        # Prepare environments
        environments, split_tensors = prepare_environments(cfg)
        train_env = environments['train']
        val_env = environments['val'] 
        test_env = environments['test']
        
        # Get observation shape
        task = train_env.sample_task()
        train_env.set_task(task)
        obs_shape = train_env.reset().shape
        
        # Create models
        encoder, policy = create_models(cfg, obs_shape)
        
        # Create trainer
        trainer = PPOTrainer(env=train_env, policy=policy, vae=encoder, config=cfg)
        
        # Training tracking
        best_val_reward = float('-inf')
        episodes_trained = 0
        
        logger.info(f"Starting training: {cfg.exp_name}")
        logger.info(f"Asset class: {cfg.asset_class}, Encoder: {cfg.encoder}, Seed: {cfg.seed}")
        logger.info(f"DSR params: eta={getattr(cfg, 'eta', 0.05)}, rf_rate={getattr(cfg, 'rf_rate', 0.02)}, tx_cost={getattr(cfg, 'transaction_cost_rate', 0.001)}")
        
        # Training loop with enhanced comprehensive tracking
        while episodes_trained < cfg.max_episodes:
            # Sample new task
            task = train_env.sample_task()
            train_env.set_task(task)
            
            # Train episodes on this task
            for _ in range(cfg.episodes_per_task):
                if episodes_trained >= cfg.max_episodes:
                    break
                    
                # Training step
                result = trainer.train_episode()
                episodes_trained += 1
                
                # === CORE TRAINING METRICS (existing) ===
                mlflow.log_metric("episode_reward", result.get('episode_reward', 0), step=episodes_trained)
                mlflow.log_metric("policy_loss", result.get('policy_loss', 0), step=episodes_trained)
                mlflow.log_metric("vae_loss", result.get('vae_loss', 0), step=episodes_trained)
                mlflow.log_metric("total_steps", result.get('total_steps', 0), step=episodes_trained)
                
                # === NEW: EPISODE-LEVEL PORTFOLIO METRICS ===
                mlflow.log_metric("episode_final_capital", result.get('episode_final_capital', 0), step=episodes_trained)
                mlflow.log_metric("episode_total_return", result.get('episode_total_return', 0), step=episodes_trained)
                mlflow.log_metric("episode_total_excess_return", result.get('episode_total_excess_return', 0), step=episodes_trained)
                mlflow.log_metric("episode_avg_concentration", result.get('episode_avg_concentration', 0), step=episodes_trained)
                mlflow.log_metric("episode_max_concentration", result.get('episode_max_concentration', 0), step=episodes_trained)
                mlflow.log_metric("episode_avg_active_positions", result.get('episode_avg_active_positions', 0), step=episodes_trained)
                mlflow.log_metric("episode_avg_cash_position", result.get('episode_avg_cash_position', 0), step=episodes_trained)
                mlflow.log_metric("episode_total_transaction_costs", result.get('episode_total_transaction_costs', 0), step=episodes_trained)
                mlflow.log_metric("episode_avg_turnover", result.get('episode_avg_turnover', 0), step=episodes_trained)
                mlflow.log_metric("episode_volatility", result.get('episode_volatility', 0), step=episodes_trained)
                mlflow.log_metric("episode_excess_volatility", result.get('episode_excess_volatility', 0), step=episodes_trained)
                
                # === NEW: DSR STATE TRACKING ===
                mlflow.log_metric("episode_final_dsr_alpha", result.get('episode_final_dsr_alpha', 0), step=episodes_trained)
                mlflow.log_metric("episode_final_dsr_beta", result.get('episode_final_dsr_beta', 0), step=episodes_trained)
                mlflow.log_metric("episode_dsr_variance", result.get('episode_dsr_variance', 0), step=episodes_trained)
                
                # === NEW: PORTFOLIO COMPOSITION TRACKING ===
                mlflow.log_metric("episode_long_exposure", result.get('episode_long_exposure', 0), step=episodes_trained)
                mlflow.log_metric("episode_short_exposure", result.get('episode_short_exposure', 0), step=episodes_trained)
                
                # === NEW: ROLLING STATISTICS ===
                mlflow.log_metric("rolling_avg_episode_reward", result.get('rolling_avg_episode_reward', 0), step=episodes_trained)
                mlflow.log_metric("rolling_std_episode_reward", result.get('rolling_std_episode_reward', 0), step=episodes_trained)
                mlflow.log_metric("rolling_avg_policy_loss", result.get('rolling_avg_policy_loss', 0), step=episodes_trained)
                mlflow.log_metric("rolling_avg_vae_loss", result.get('rolling_avg_vae_loss', 0), step=episodes_trained)
                
                # === NEW: TRAINING DIAGNOSTICS ===
                mlflow.log_metric("num_episodes_in_batch", result.get('num_episodes_in_batch', 1), step=episodes_trained)
                mlflow.log_metric("steps_per_episode", result.get('steps_per_episode', 0), step=episodes_trained)
                
                # === NEW: VAE LOSS COMPONENTS (when available) ===
                vae_components = {k: v for k, v in result.items() if k.startswith('vae_') and k != 'vae_loss'}
                for component_name, component_value in vae_components.items():
                    mlflow.log_metric(component_name, component_value, step=episodes_trained)
                
                # === NEW: STEP-BY-STEP DATA AS ARTIFACTS (every N episodes) ===
                if episodes_trained % 50 == 0 and 'step_data' in result:
                    step_data = result['step_data']
                    
                    # Create artifact with step-by-step data
                    step_data_file = f"episode_{episodes_trained}_step_data.json"
                    
                    # Convert numpy arrays to lists for JSON serialization
                    serializable_step_data = {}
                    for key, value in step_data.items():
                        if isinstance(value, list):
                            # Handle list of arrays (like step_weights)
                            if value and isinstance(value[0], np.ndarray):
                                serializable_step_data[key] = [arr.tolist() for arr in value]
                            else:
                                serializable_step_data[key] = value
                        elif isinstance(value, np.ndarray):
                            serializable_step_data[key] = value.tolist()
                        else:
                            serializable_step_data[key] = value
                    
                    # Save and log artifact
                    with open(step_data_file, 'w') as f:
                        json.dump(serializable_step_data, f, indent=2)
                    mlflow.log_artifact(step_data_file, "step_data")
                    os.unlink(step_data_file)  # Cleanup temp file
                    
                    logger.info(f"Logged step-by-step data artifact for episode {episodes_trained}")
                
                # === NEW: DETAILED VAE LOGGING (every 10 episodes when VAE is active) ===
                if episodes_trained % 10 == 0 and result.get('vae_loss', 0) > 0:
                    # Log additional VAE insights
                    if 'vae_latent_effective_dim' in result:
                        mlflow.log_metric("vae_latent_utilization", 
                                        result.get('vae_latent_effective_dim', 0) / max(result.get('vae_latent_dim', 1), 1), 
                                        step=episodes_trained)
                    
                    if 'vae_kl_weight_ratio' in result:
                        mlflow.log_metric("vae_kl_dominance", result.get('vae_kl_weight_ratio', 0), step=episodes_trained)
                
                # === NEW: PORTFOLIO RISK METRICS (every 25 episodes) ===
                if episodes_trained % 25 == 0:
                    # Calculate additional risk metrics from recent episodes
                    recent_returns = [result.get('episode_total_return', 0)]  # Would need to track more episodes
                    recent_volatilities = [result.get('episode_volatility', 0)]
                    
                    if len(recent_returns) > 1:
                        mlflow.log_metric("recent_return_skewness", float(np.mean(recent_returns)), step=episodes_trained)
                        mlflow.log_metric("recent_volatility_avg", float(np.mean(recent_volatilities)), step=episodes_trained)
                
                # Validation
                if episodes_trained % cfg.val_interval == 0:
                    val_results = evaluate(val_env, policy, encoder, cfg, cfg.val_episodes)
                    
                    # === ENHANCED VALIDATION LOGGING ===
                    # Core validation metrics (existing)
                    for key, value in val_results.items():
                        mlflow.log_metric(f"val_{key}", value, step=episodes_trained)
                    
                    # === NEW: VALIDATION INSIGHTS ===
                    current_val_reward = val_results['avg_reward']
                    
                    # Track validation improvement
                    val_improvement = current_val_reward - best_val_reward if 'best_val_reward' in locals() else 0
                    mlflow.log_metric("val_reward_improvement", val_improvement, step=episodes_trained)
                    
                    # Validation stability metrics
                    if 'std_reward' in val_results:
                        val_stability = val_results['avg_reward'] / max(val_results['std_reward'], 1e-8)
                        mlflow.log_metric("val_reward_stability", val_stability, step=episodes_trained)
                    
                    # Risk-adjusted validation performance
                    if 'avg_return' in val_results and 'avg_volatility' in val_results:
                        val_risk_adjusted = val_results['avg_return'] / max(val_results['avg_volatility'], 1e-8)
                        mlflow.log_metric("val_risk_adjusted_return", val_risk_adjusted, step=episodes_trained)
                    
                    # Track best model
                    if current_val_reward > best_val_reward:
                        best_val_reward = current_val_reward
                        
                        # Enhanced best model tracking
                        mlflow.log_metric("best_val_reward", best_val_reward, step=episodes_trained)
                        mlflow.log_metric("best_val_episode", episodes_trained, step=episodes_trained)
                        
                        # Log best model portfolio characteristics
                        mlflow.log_metric("best_model_concentration", result.get('episode_avg_concentration', 0), step=episodes_trained)
                        mlflow.log_metric("best_model_turnover", result.get('episode_avg_turnover', 0), step=episodes_trained)
                        mlflow.log_metric("best_model_cash_allocation", result.get('episode_avg_cash_position', 0), step=episodes_trained)
                    
                    logger.info(f"Episode {episodes_trained}: val_reward={current_val_reward:.4f}, best={best_val_reward:.4f}")
                    
                    # === NEW: EARLY STOPPING DIAGNOSTICS ===
                    episodes_since_best = episodes_trained - mlflow.get_metric("best_val_episode") if "best_val_episode" in locals() else 0
                    mlflow.log_metric("episodes_since_best", episodes_since_best, step=episodes_trained)
                    
                    # Overfitting detection
                    train_val_gap = result.get('rolling_avg_episode_reward', 0) - current_val_reward
                    mlflow.log_metric("train_val_gap", train_val_gap, step=episodes_trained)
                
                # === NEW: MEMORY AND PERFORMANCE MONITORING ===
                if episodes_trained % 100 == 0:
                    import psutil
                    import gc
                    
                    # Memory usage
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    mlflow.log_metric("memory_usage_mb", memory_mb, step=episodes_trained)
                    
                    # GPU memory if available
                    if torch.cuda.is_available():
                        gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                        mlflow.log_metric("gpu_memory_mb", gpu_memory_mb, step=episodes_trained)
                        
                        # GPU utilization
                        gpu_utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                        mlflow.log_metric("gpu_utilization_pct", gpu_utilization, step=episodes_trained)
                    
                    # Training speed
                    training_speed = episodes_trained / max((time.time() - trainer.training_start_time).total_seconds() / 3600, 1e-6)
                    mlflow.log_metric("episodes_per_hour", training_speed, step=episodes_trained)
                    
                    # Force garbage collection for memory management
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Final test evaluation and backtest
        logger.info("Running final evaluation and backtest...")
        test_results = evaluate(test_env, policy, encoder, cfg, cfg.test_episodes)

        # Running backtest
        logger.info("Running sequential backtest...")
        backtest_results = backtest_results = run_sequential_backtest(datasets, policy, encoder, cfg, split='test')
        
        # Log test and backtest results
        for key, value in test_results.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"test_{key}", value)
        
        for key, value in backtest_results.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"backtest_{key}", value)
        
        # Save models to MinIO via MLflow
        model_dict = {
            'policy': policy,
            'encoder': encoder
        }
        
        # Log essential artifacts (models + config) to MinIO
        log_essential_artifacts(model_dict, cfg.__dict__, cfg.exp_name)
        
        # Also save locally for backup
        model_dir = Path("models") / cfg.exp_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if encoder is not None:
            torch.save(encoder.state_dict(), model_dir / "encoder.pt")
        
        torch.save(policy.state_dict(), model_dir / "policy.pt")
        
        # Final results
        final_results = {
            "episodes_trained": episodes_trained,
            "best_val_reward": best_val_reward,
            "final_test_reward": test_results['avg_reward'],
            "backtest_sharpe": backtest_results['sharpe_ratio'],
            "backtest_return": backtest_results['total_return'],
            "backtest_max_drawdown": backtest_results['max_drawdown'],
            "training_completed": True
        }
        
        logger.info(f"Training completed: {cfg.exp_name}")
        logger.info(f"Final test reward: {test_results['avg_reward']:.4f}")
        logger.info(f"Backtest Sharpe: {backtest_results['sharpe_ratio']:.4f}")
        
        return final_results
        
    except Exception as e:
        logger.error(f"Training failed for {cfg.exp_name}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Log failure
        mlflow.log_metric("training_completed", 0)
        mlflow.log_param("error_message", str(e))
        
        return {
            "training_completed": False,
            "error": str(e),
            "episodes_trained": episodes_trained if 'episodes_trained' in locals() else 0
        }
    
    finally:
        # Cleanup
        cleanup_gpu_memory()

def run_experiment_batch(experiments, experiment_name: str = "test_001"):
    """Run batch of experiments using ExperimentManager (simplified without resource management)."""
    
    # Create experiment manager without resource limits (since the current ExperimentManager doesn't support it)
    manager = ExperimentManager(
        experiments, 
        max_retries=0
    )
    
    # Run all experiments
    summary = manager.run_all_experiments(experiment_name)
    
    return summary


def ensure_mlflow_setup():
    """Ensure MLflow is properly configured."""
    from mlflow_logger import setup_mlflow
    return setup_mlflow()


# TODO delete when confirmed useless
# def log_essential_artifacts(model_dict, config_dict, experiment_name):
#     """Log essential artifacts - placeholder for now."""
#     try:
#         from mlflow_setup import log_essential_artifacts as log_fn
#         log_fn(model_dict, config_dict, experiment_name)
#     except ImportError:
#         logger.warning("Could not import mlflow_setup - skipping artifact logging")

def main():
    """Main experiment runner."""
    
    # Setup MLflow before anything else
    logger.info("Setting up MLflow configuration...")
    backend = ensure_mlflow_setup()
    logger.info(f"MLflow configured with {backend} backend")
    
    # Generate all experiment configurations
    experiments = generate_experiment_configs(num_seeds=1) #TODO experiments = generate_experiment_configs(num_seeds=10)
    
    logger.info(f"Generated {len(experiments)} experiment configurations")
    logger.info("Experiment matrix:")
    logger.info("- Asset classes: SP500, Crypto")
    logger.info("- Encoders: VAE, None, HMM")
    logger.info("- Seeds: 0-9 (10 seeds per combination)")
    logger.info(f"- Total: {len(experiments)} experiments")
    
    # Run subset for testing first (optional)
    if test_mode:
        logger.info("TEST MODE: Running only first 2 experiments")
        experiments = experiments[:2]
    
    if debug_mode:
        logger.debug("Debug mode configuration:")
        logger.debug(f"- Logging level: DEBUG")
        logger.debug(f"- Log file: experiment_debug.log")
        logger.debug(f"- Resource monitoring: disabled (not implemented in current ExperimentManager)")
        logger.debug(f"- Checkpoint directory: experiment_checkpoints/")
    
    # Run all experiments
    summary = run_experiment_batch(experiments, experiment_name="test_study_001")
    
    if debug_mode:
        logger.debug("Final summary keys:")
        for key in summary.keys():
            logger.debug(f"  {key}: {type(summary[key])}")
    
    return summary


if __name__ == "__main__":
    main()