#import mlflow
#import mlflow.pytorch
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
import argparse
from tqdm import tqdm
from hpo_utils import load_hpo_params


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
from algorithms.pretrain_hmm import pretrain_hmm
from algorithms.trainer import PPOTrainer
from csv_logger import CSVLogger, TrainingCSVLogger, ValidationCSVLogger
# Import evaluation functions
from evaluation_backtest import evaluate, run_sequential_backtest
import shutil


def save_checkpoint(ckpt_dir: Path, state: dict):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"checkpoint_ep{state['episodes_trained']}.pt"
    torch.save(state, path)

    # Remove MLflow-specific checkpoint info
    # Just save a simple metadata file
    metadata = {
        "episodes_trained": state['episodes_trained'],
        "checkpoint_path": str(path),
        "timestamp": datetime.now().isoformat()
    }
    with open(ckpt_dir / "checkpoint_info.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"💾 Saved checkpoint: {path}")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


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
        force_recreate=cfg.force_recreate   # instead of hardcoded True
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

    if cfg.n_assets_limit is not None:
        logger.info(f"Limiting to first {cfg.n_assets_limit} assets (from {datasets['train'].num_assets})")
        
        # Filter each split
        for split_name, dataset_split in datasets.items():
            # Get first n tickers
            selected_tickers = sorted(dataset_split.tickers)[:cfg.n_assets_limit]
            
            # Filter the dataframe
            dataset_split.data = dataset_split.data[
                dataset_split.data['ticker'].isin(selected_tickers)
            ].copy()
            
            # Update metadata
            dataset_split.tickers = selected_tickers
            dataset_split.num_assets = len(selected_tickers)
            
            logger.info(f"  {split_name}: {dataset_split.num_assets} assets, {len(dataset_split.data)} rows")
    
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
            steps_per_year=steps_per_year,
            inflation_rate=cfg.inflation_rate,
            reward_type=cfg.reward_type,
            reward_lookback=cfg.reward_lookback,
            concentration_penalty=cfg.concentration_penalty,
            concentration_target=cfg.concentration_target,
            concentration_lambda=cfg.concentration_lambda,
        )
    
    return environments, split_tensors, datasets



def create_models(cfg: TrainingConfig, obs_shape) -> tuple:
    """Create encoder and policy models based on configuration."""
    device = torch.device(cfg.device)
    
    encoder = None
    if cfg.encoder == "vae":
        encoder = VAE(
            obs_dim=obs_shape,
            num_assets=cfg.num_assets,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim
        ).to(device)
        logger.info(f"✅ Created VAE encoder with latent_dim={cfg.latent_dim}")
    
    elif cfg.encoder == "hmm":
        from algorithms.pretrain_hmm import pretrain_hmm
        
        logger.info("🔄 Running HMM pre-training...")
        
        # Pass the full config to pre-training for consistency
        success, encoder_path = pretrain_hmm(
            asset_class=cfg.asset_class, 
            seed=cfg.seed,
            config=cfg  # Pass full config for parameter consistency
        )
        
        if not success:
            logger.error("❌ HMM pre-training failed")
            raise RuntimeError("HMM pre-training failed")
        
        # Create encoder with correct dimensions (4 states for HMM)
        hmm_latent_dim = 4  # Override config for HMM
        encoder = HMMEncoder(
            obs_dim=obs_shape,
            num_assets=cfg.num_assets,
            latent_dim=hmm_latent_dim,
            hidden_dim=cfg.hidden_dim
        ).to(device)
        
        # Load pre-trained weights
        if encoder_path and Path(encoder_path).exists():
            encoder.load_state_dict(torch.load(encoder_path, map_location=device))
            logger.info(f"📥 Loaded pre-trained HMM weights from {encoder_path}")
        else:
            logger.warning(f"⚠️ Pre-trained weights not found, using random weights")
        
        # Update cfg.latent_dim to match HMM for policy creation
        cfg.latent_dim = hmm_latent_dim
        
        logger.info(f"✅ Created HMM encoder with {hmm_latent_dim} states (pre-trained)")
    else:
        logger.info("✅ No encoder (encoder=none)")
    
    # Create policy
    policy = PortfolioPolicy(
        obs_shape=obs_shape,
        latent_dim=cfg.latent_dim,
        num_assets=cfg.num_assets,
        hidden_dim=cfg.hidden_dim,
        noise_factor=cfg.noise_factor,
        random_policy=cfg.random_policy,
        long_only=cfg.long_only
    ).to(device)
    
    logger.info(f"✅ Created policy with obs_shape={obs_shape}, latent_dim={cfg.latent_dim}")
    
    return encoder, policy

def run_training(cfg: TrainingConfig) -> Dict[str, Any]:
    """Run complete training pipeline with checkpoints + MLflow resume."""

    # if mlflow.active_run():
    #     mlflow.end_run()

    ckpt_dir = Path(cfg.exp_name) / cfg.encoder / cfg.asset_class / "checkpoints"

    resume_state = None
    episodes_trained, best_val_reward = 0, float("-inf")

    training_logger = TrainingCSVLogger(cfg.exp_name, cfg.seed, cfg.asset_class, cfg.encoder)
    validation_logger = ValidationCSVLogger(cfg.exp_name, cfg.seed, cfg.asset_class, cfg.encoder)


    # === Resume or start fresh ===
    if getattr(cfg, "force_recreate", False):
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
            logger.info(f"🗑️ Removed old checkpoints for {cfg.exp_name}")
        
        # Also remove log files
        log_dir = Path(cfg.encoder) / cfg.asset_class / "experiment_logs"
        if log_dir.exists():
            for log_file in log_dir.glob(f"{cfg.exp_name}*"):
                log_file.unlink()
            logger.info(f"🗑️ Removed old logs for {cfg.exp_name}")

    # Always start fresh
    resume_state = None
    episodes_trained = 0
    best_val_reward = float("-inf")
    patience_counter = 0

    # === NEW: Initialize MLflow integration ===
    # from mlflow_logger import MLflowIntegration
    # mlflow_integration = MLflowIntegration(run_name=cfg.exp_name, config=vars(cfg))
    # mlflow_integration.setup_mlflow()
    # mlflow_integration.log_config()
    # csv_logger = CSVLogger(run_name=cfg.exp_name, config=vars(cfg))
    # csv_logger.setup_mlflow()
    # csv_logger.log_config()

    try:
        # Setup
        seed_everything(cfg.seed)

        # Ensure dataset exists
        cfg.data_path = ensure_dataset_exists(cfg)

        # Apply HPO parameters if provided
        if hasattr(cfg, '_from_hpo_path'):
            cfg = load_hpo_params(cfg._from_hpo_path, cfg)

        # Prepare environments
        environments, split_tensors, datasets = prepare_environments(cfg)
        train_env, val_env, test_env = environments['train'], environments['val'], environments['test']

        # === Define observation shape depending on encoder type === TODO
        task = train_env.sample_task()
        train_env.set_task(task)
        obs_shape = train_env.reset().shape

        # Create models
        encoder, policy = create_models(cfg, obs_shape)
        logger.info(f"=== Models Created ===")
        logger.info(f"  Encoder type: {type(encoder).__name__ if encoder else 'None'}")
        logger.info(f"  Config encoder: {cfg.encoder}")
        logger.info(f"  Config disable_vae: {cfg.disable_vae}")

        # Create trainer
        trainer = PPOTrainer(env=train_env, policy=policy, vae=encoder, config=cfg)

        # === Apply resume state after objects exist ===
        if resume_state:
            policy.load_state_dict(resume_state["policy"])
            if encoder and resume_state.get("encoder"):
                encoder.load_state_dict(resume_state["encoder"])
            
            # Load optimizers (handle both old and new checkpoint formats)
            if "policy_optimizer" in resume_state:
                trainer.policy_optimizer.load_state_dict(resume_state["policy_optimizer"])
                if trainer.vae_optimizer and resume_state.get("vae_optimizer"):
                    trainer.vae_optimizer.load_state_dict(resume_state["vae_optimizer"])
            elif "optimizer" in resume_state:
                # Old checkpoint format - load into policy optimizer only
                trainer.policy_optimizer.load_state_dict(resume_state["optimizer"])
            
            torch.set_rng_state(resume_state["torch_rng"])
            np.random.set_state(tuple(resume_state["numpy_rng"]))
            import random
            random.setstate(tuple(resume_state["py_rng"]))
            logger.info("✅ Restored model, optimizer(s), and RNG state")

        # corrected loop now with task appending
        logger.info(f"Starting training: {cfg.exp_name}")
        logger.info(f"Asset class: {cfg.asset_class}, Encoder: {cfg.encoder}, Seed: {cfg.seed}")
        logger.info(f"Training structure: {cfg.max_episodes // cfg.episodes_per_task} tasks × {cfg.episodes_per_task} episodes/task")

        tasks_trained = 0
        episodes_trained = 0
        early_stopped = False

        total_tasks = cfg.max_episodes // cfg.episodes_per_task

        with tqdm(total=total_tasks, desc=f"Training Progress (tasks)") as pbar:
            while episodes_trained < cfg.max_episodes and not early_stopped:
                
                # Train on one task (multiple episodes with persistent context) TODO
                result = trainer.train_on_task()
                episodes_trained += cfg.episodes_per_task
                tasks_trained += 1
                pbar.update(1)
                
                # logger.info(f"=== main.py train_on_task RESULT DEBUG ===")
                # logger.info(f"  Task: {tasks_trained}/{total_tasks}")
                # logger.info(f"  Episodes trained: {episodes_trained}")
                # logger.info(f"  Result keys: {list(result.keys())}")
                # logger.info(f"  Result values sample:")
                # for k, v in result.items():
                #     if isinstance(v, (int, float, str, bool)):
                #         logger.info(f"    {k}: {v}")
                #     elif isinstance(v, list) and len(v) <= 5:
                #         logger.info(f"    {k}: {v}")
                #     else:
                #         logger.info(f"    {k}: {type(v)} (len={len(v) if hasattr(v, '__len__') else 'N/A'})")

                # Log to CSV
                training_logger.log_task(tasks_trained, result)
                
                # Checkpoint every 10 tasks
                if tasks_trained % 10 == 0:
                    checkpoint_state = {
                        "episodes_trained": episodes_trained,
                        "tasks_trained": tasks_trained,
                        "best_val_reward": best_val_reward,
                        "policy": policy.state_dict(),
                        "encoder": encoder.state_dict() if encoder else None,
                        "policy_optimizer": trainer.policy_optimizer.state_dict(),
                        "vae_optimizer": trainer.vae_optimizer.state_dict() if trainer.vae_optimizer else None,
                        "torch_rng": torch.get_rng_state(),
                        "numpy_rng": np.random.get_state(),
                        "py_rng": __import__("random").getstate(),
                    }
                    # save_checkpoint(ckpt_dir, checkpoint_state)
                
                # Validation (every N episodes worth)
                if episodes_trained % cfg.val_interval == 0:
                    val_results = evaluate(val_env, policy, encoder, cfg, "validation", cfg.val_episodes)
                    validation_logger.log_validation(episodes_trained, val_results)
                    
                    current_val_reward = val_results.get("validation: avg_reward", -1e9)
                    logger.info(f"Validation at episode {episodes_trained}: {current_val_reward}")
                    
                    if current_val_reward > best_val_reward:
                        best_val_reward = current_val_reward
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if (episodes_trained >= cfg.min_episodes_before_stopping and 
                        patience_counter >= cfg.early_stopping_patience):
                        logger.info(f"Early stopping triggered at episode {episodes_trained}")
                        early_stopped = True
                        break
        
        # Final evaluation & backtest
        logger.info("Running final evaluation and backtest...")
        test_results = evaluate(test_env, policy, encoder, cfg, "test", cfg.test_episodes)
        backtest_results = run_sequential_backtest(datasets, policy, encoder, cfg, split='test')

        #mlflow_integration.log_validation_results(episodes_trained, test_results)
        #mlflow_integration.log_backtest_results(backtest_results)

        # Save final models + config
        model_dict = {"policy": policy, "encoder": encoder}
        #mlflow_integration.log_essential_artifacts(model_dict, vars(cfg), cfg.exp_name)
        #mlflow_integration.log_final_summary(True, episodes_trained)

        #validation_logger.log_validation(episodes_trained, test_results)    

        final_results = {
            "episodes_trained": episodes_trained,
            "best_val_reward": best_val_reward,
            "final_test_reward": test_results['avg_reward'],
            "backtest_sharpe": backtest_results['backtest_sharpe'],
            "backtest_return": backtest_results['total_return'],
            "backtest_max_drawdown": backtest_results['max_drawdown'],
            "training_completed": True,
        }

        logger.info(f"Training completed: {cfg.exp_name}")
        return final_results

    except Exception as e:
        logger.error(f"Training failed for {cfg.exp_name}: {str(e)}")
        logger.error(traceback.format_exc())
        #mlflow_integration.log_final_summary(False, episodes_trained, error_msg=str(e))
        #csv_logger.log_final_summary(False, episodes_trained, error_msg=str(e))
        return {
            "training_completed": False,
            "error": str(e),
            "episodes_trained": episodes_trained,
        }

    finally:
        cleanup_gpu_memory()
        #mlflow.end_run()


def run_experiment_batch(experiments, experiment_name: str = "test_001", force_recreate: bool = False):
    """Run batch of experiments using ExperimentManager (simplified without resource management)."""
    
    # Create experiment manager without resource limits (since the current ExperimentManager doesn't support it)
    manager = ExperimentManager(
        experiments, 
        max_retries=0,
        force_recreate=force_recreate
    )    
    return manager.run_all_experiments(experiment_name)



def main():
    """Main experiment runner."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=None, help="Override experiment name")
    parser.add_argument("--force_recreate", action="store_true", help="Force dataset and MLflow overwrite")
    parser.add_argument("--encoder", type=str, choices=["vae", "hmm", "none"], help="Encoder type to run")
    parser.add_argument("--datatype", type=str, choices=["sp500", "crypto"], help="Dataset type to run")
    parser.add_argument("--reward_type", type=str, choices=["dsr", "sharpe", "drawdown"], 
                    default="dsr", help="Reward function type")
    #parser.add_argument("--reward_lookback", type=int, default=20, 
                    #help="Lookback window for Sharpe/Drawdown calculation")
    parser.add_argument("--disable_transaction_costs", action="store_true", 
                    help="Disable transaction costs (set to 0)")
    parser.add_argument("--transaction_cost_rate", type=float, default=None,
                    help="Override transaction cost rate (default: 0.001)")
    parser.add_argument("--disable_inflation", action="store_true",
                    help="Disable inflation penalty (set to 0)")
    parser.add_argument("--inflation_rate", type=float, default=None,
                    help="Override inflation rate (default: 0.1)")
    parser.add_argument("--load_hpo_params", type=str, default=None,
                    help="Path to HPO results JSON")
    parser.add_argument("--n_assets", type=int, default=None,
                        help="Number of assets to use (default: all 30)")
    parser.add_argument("--enable_concentration_penalty", action="store_true",
                        help="Enable concentration penalty to encourage diversification")
    parser.add_argument("--long_only", action="store_true",  # ADD THIS
                        help="Enable long-only portfolio (disable short positions)")              

    args = parser.parse_args()

    # Setup MLflow before anything else
    # logger.info("Setting up MLflow configuration...")
    # backend = ensure_mlflow_setup()
    # logger.info(f"MLflow configured with {backend} backend")
    logger.info("Using CSV logging backend")

    # Generate all experiment configurations
    experiments = generate_experiment_configs(num_seeds=5) # TODO 
    # Apply CLI overrides
    # Apply CLI overrides
    for exp in experiments:
        if args.exp_name:
            exp.exp_name = args.exp_name
        if args.force_recreate:
            exp.force_recreate = True
        
        # Store HPO path for later application
        if args.load_hpo_params:
            exp._hpo_path = args.load_hpo_params
        
        if args.n_assets is not None:
            exp.n_assets = args.n_assets
        
        # Cost/inflation overrides
        if args.disable_transaction_costs:
            exp.transaction_cost_rate = 0.0
        elif args.transaction_cost_rate is not None:
            exp.transaction_cost_rate = args.transaction_cost_rate
            
        if args.disable_inflation:
            exp.inflation_rate = 0.0
        elif args.inflation_rate is not None:
            exp.inflation_rate = args.inflation_rate

        if args.enable_concentration_penalty:
            exp.concentration_penalty = True
        
        if args.long_only:
            exp.long_only = True
            
    if args.encoder:
        experiments = [exp for exp in experiments if exp.encoder == args.encoder]
    if args.datatype:
        experiments = [exp for exp in experiments if exp.asset_class == args.datatype]

    # logger.info(f"Generated {len(experiments)} experiment configurations")
    # logger.info("Experiment matrix:")
    # logger.info("- Asset classes: SP500, Crypto")
    # logger.info("- Encoders: VAE, None, HMM")
    # logger.info("- Seeds: 0-9 (10 seeds per combination)")
    # logger.info(f"- Total: {len(experiments)} experiments")
    
    # Store HPO path in experiments
    if args.load_hpo_params:
        for exp in experiments:
            exp._hpo_params_path = args.load_hpo_params

    # Run all experiments
    summary = run_experiment_batch(
        experiments, 
        experiment_name=args.exp_name or "test_study",
        force_recreate=args.force_recreate
    )

    
    if debug_mode:
        logger.debug("Final summary keys:")
        for key in summary.keys():
            logger.debug(f"  {key}: {type(summary[key])}")
    
    return summary


if __name__ == "__main__":
    main()



# full studies 
# t1: sp500 DSR - done
# t2: sp500 Sharpe - done

# t1: crypto Sharpe HPO
# t2: crypto 



# t1: sp500 drawdown - running on new instance
# t2 crypto drawdown


