#!/usr/bin/env python3
"""
Simplified VariBAD Portfolio Optimization Pipeline
Works with the current project structure
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# Setup logging
def setup_logging(log_level="INFO"):
    """Setup comprehensive logging for the pipeline."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"varibad_pipeline_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Pipeline started - logs saved to {log_file}")
    return logger

def check_dependencies():
    """Check if all required packages are installed with correct import names."""
    required_packages = [
        'pandas', 'numpy', 'torch', 'yfinance', 'sklearn',
        'matplotlib', 'seaborn', 'gym'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print("Install with:")
        for pkg in missing:
            if pkg == 'sklearn':
                print(f"  pip install scikit-learn  # imports as 'sklearn'")
            else:
                print(f"  pip install {pkg}")
        sys.exit(1)
    
    print("✅ All dependencies satisfied")

def create_directory_structure():
    """Create necessary directories for the pipeline."""
    directories = [
        "data", "logs", "checkpoints", "results", "plots"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    
    print("✅ Directory structure created")

def download_and_clean_data(logger):
    """Simplified data processing pipeline."""
    logger.info("Starting data processing pipeline...")
    
    final_path = "data/sp500_rl_ready_cleaned.parquet"
    
    # Check if data already exists
    if os.path.exists(final_path):
        logger.info(f"✅ Data already exists at {final_path}")
        
        # Validate existing data
        try:
            df = pd.read_parquet(final_path)
            nan_count = df.isnull().sum().sum()
            
            logger.info(f"✅ Data validation complete!")
            logger.info(f"   Dataset shape: {df.shape}")
            logger.info(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
            logger.info(f"   Tickers: {df['ticker'].nunique()}")
            logger.info(f"   Features: {len(df.columns)}")
            logger.info(f"   Missing values: {nan_count}")
            
            return final_path
            
        except Exception as e:
            logger.warning(f"Existing data file is corrupted: {e}. Will recreate.")
    
    # Create new data
    logger.info("Creating new dataset from scratch...")
    
    try:
        # Import the data pipeline
        from varibad.data_pipeline import create_rl_dataset
        
        # Create the dataset
        final_path = create_rl_dataset()
        
        logger.info(f"✅ Data processing complete! Dataset saved to: {final_path}")
        return final_path
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def train_varibad_model(data_path, config, logger):
    """Train the VariBAD model."""
    logger.info("Starting VariBAD training...")
    
    try:
        from varibad.core.trainer import VariBADTrainer
        
        # Initialize trainer
        trainer = VariBADTrainer(
            data_path=data_path,
            episode_length=config.episode_length,
            action_dim=30,  # Your 30 S&P 500 companies
            latent_dim=config.latent_dim,
            max_episodes_buffer=config.buffer_size,
            enable_short_selling=config.short_selling,
            device=config.device
        )
        
        logger.info(f"✅ Trainer initialized")
        logger.info(f"   State dimension: {trainer.state_dim}")
        logger.info(f"   Action dimension: {trainer.actual_action_dim}")
        logger.info(f"   Model parameters: {sum(p.numel() for p in trainer.varibad.parameters()):,}")
        
        # Train model
        stats = trainer.train(
            num_iterations=config.num_iterations,
            episodes_per_iteration=config.episodes_per_iteration,
            vae_updates_per_iteration=config.vae_updates,
            eval_frequency=config.eval_frequency,
            save_frequency=config.save_frequency
        )
        
        # Save final model
        checkpoint_path = f"checkpoints/varibad_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        trainer.save_checkpoint(checkpoint_path)
        
        logger.info(f"✅ Training complete! Model saved to {checkpoint_path}")
        return checkpoint_path, stats
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def evaluate_model(checkpoint_path, data_path, logger):
    """Evaluate a trained model."""
    logger.info(f"Evaluating model from {checkpoint_path}")
    
    try:
        from varibad.core.trainer import VariBADTrainer
        import torch
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        
        # Recreate trainer
        trainer = VariBADTrainer(
            data_path=data_path,
            state_dim=config['state_dim'],
            action_dim=config['actual_action_dim'] // 2,
            device='cpu'
        )
        
        # Load model weights
        trainer.varibad.load_state_dict(checkpoint['varibad_state_dict'])
        
        # Run evaluation
        eval_stats = trainer.evaluate(num_episodes=50)
        
        logger.info("✅ Evaluation complete!")
        for key, value in eval_stats.items():
            logger.info(f"   {key}: {value:.4f}")
        
        return eval_stats
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def main():
    """Main pipeline entry point."""
    parser = argparse.ArgumentParser(description="VariBAD Portfolio Optimization Pipeline")
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                      choices=['data_only', 'train', 'resume', 'evaluate'],
                      help='Pipeline mode to run')
    
    # Training parameters
    parser.add_argument('--num_iterations', type=int, default=1000,
                      help='Number of training iterations')
    parser.add_argument('--episode_length', type=int, default=30,
                      help='Length of each trading episode')
    parser.add_argument('--episodes_per_iteration', type=int, default=5,
                      help='Episodes to collect per iteration')
    parser.add_argument('--vae_updates', type=int, default=10,
                      help='VAE updates per iteration')
    parser.add_argument('--latent_dim', type=int, default=5,
                      help='Latent dimension for VariBAD')
    parser.add_argument('--buffer_size', type=int, default=500,
                      help='Max episodes in trajectory buffer')
    
    # Model parameters
    parser.add_argument('--short_selling', action='store_true',
                      help='Enable short selling')
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to use (cpu/cuda/auto)')
    
    # Evaluation parameters
    parser.add_argument('--eval_frequency', type=int, default=50,
                      help='Evaluate every N iterations')
    parser.add_argument('--save_frequency', type=int, default=100,
                      help='Save checkpoint every N iterations')
    
    # File paths
    parser.add_argument('--checkpoint', type=str,
                      help='Path to checkpoint for resume/evaluate modes')
    parser.add_argument('--data_path', type=str, default='data/sp500_rl_ready_cleaned.parquet',
                      help='Path to processed dataset')
    
    # Logging
    parser.add_argument('--log_level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      help='Logging level')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.log_level)
    check_dependencies()
    create_directory_structure()
    
    # Auto-detect device
    if args.device == 'auto':
        import torch
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Auto-detected device: {args.device}")
    
    logger.info(f"🚀 Starting VariBAD pipeline in '{args.mode}' mode")
    logger.info(f"Configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    
    try:
        # Execute based on mode
        if args.mode == 'data_only':
            logger.info("📊 Data processing mode")
            final_data_path = download_and_clean_data(logger)
            logger.info(f"✅ Data processing complete! Dataset ready at: {final_data_path}")
        
        elif args.mode == 'train':
            logger.info("🏋️ Training mode")
            # Process data first
            final_data_path = download_and_clean_data(logger)
            # Then train
            checkpoint_path, stats = train_varibad_model(final_data_path, args, logger)
            logger.info(f"✅ Training complete! Model saved to: {checkpoint_path}")
        
        elif args.mode == 'resume':
            logger.info("🔄 Resume training mode")
            if not args.checkpoint:
                logger.error("--checkpoint required for resume mode")
                sys.exit(1)
            
            # Ensure data exists
            if not os.path.exists(args.data_path):
                logger.info("Data not found, processing first...")
                args.data_path = download_and_clean_data(logger)
            
            logger.warning("Resume functionality not yet implemented")
        
        elif args.mode == 'evaluate':
            logger.info("📊 Evaluation mode")
            if not args.checkpoint:
                logger.error("--checkpoint required for evaluate mode")
                sys.exit(1)
            
            # Ensure data exists
            if not os.path.exists(args.data_path):
                logger.info("Data not found, processing first...")
                args.data_path = download_and_clean_data(logger)
            
            eval_stats = evaluate_model(args.checkpoint, args.data_path, logger)
            logger.info("✅ Evaluation complete!")
    
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    logger.info("🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()