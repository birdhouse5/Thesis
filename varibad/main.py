#!/usr/bin/env python3
"""
VariBAD Portfolio Optimization Pipeline - Flat Structure Version
Improved with better import handling and error messages
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
    logger.info(f"VariBAD Pipeline started - logs saved to {log_file}")
    logger.info(f"Running from: {os.getcwd()}")
    logger.info(f"Script location: {Path(__file__).absolute()}")
    return logger

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = [
        'pandas', 'numpy', 'torch', 'yfinance', 'sklearn',
        'matplotlib', 'seaborn', 'gym'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print("Install with:")
        for pkg in missing:
            if pkg == 'sklearn':
                print(f"  pip install scikit-learn")
            else:
                print(f"  pip install {pkg}")
        sys.exit(1)
    
    print("✅ All dependencies satisfied")

def create_directory_structure():
    """Create necessary directories for the pipeline."""
    directories = [
        "data", "logs", "checkpoints", "results", "plots", "config"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    
    print("✅ Directory structure created")

def download_and_clean_data(logger):
    """Data processing pipeline with improved error handling."""
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
        # Import the data pipeline with proper error handling
        logger.info("Attempting to import data pipeline...")
        
        # Try multiple import paths
        data_pipeline_module = None
        import_attempts = [
            "varibad.data_pipeline",
            "data_pipeline",
            "varibad.data_pipeline"
        ]
        
        for import_path in import_attempts:
            try:
                if import_path == "varibad.data_pipeline":
                    from varibad.data_pipeline import create_rl_dataset
                elif import_path == "data_pipeline":
                    from data_pipeline import create_rl_dataset
                
                data_pipeline_module = True
                logger.info(f"✅ Successfully imported from {import_path}")
                break
                
            except ImportError as e:
                logger.debug(f"Failed to import from {import_path}: {e}")
                continue
        
        if not data_pipeline_module:
            raise ImportError("Could not import data_pipeline from any location")
        
        # Create the dataset
        final_path = create_rl_dataset()
        
        logger.info(f"✅ Data processing complete! Dataset saved to: {final_path}")
        return final_path
        
    except ImportError as e:
        logger.error(f"Failed to import data pipeline: {e}")
        logger.error("Available Python path:")
        for path in sys.path:
            logger.error(f"  {path}")
        logger.error("Make sure you're running from the project root directory")
        logger.error("Current working directory: " + os.getcwd())
        raise
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def train_varibad_model(data_path, config, logger):
    """Train the VariBAD model with comprehensive error handling."""
    logger.info("Starting VariBAD training...")
    
    try:
        # Import with detailed error handling
        logger.info("Attempting to import VariBAD trainer...")
        
        try:
            from varibad.core.trainer import VariBADTrainer
            logger.info("✅ Successfully imported VariBADTrainer")
        except ImportError as e:
            logger.error(f"Failed to import VariBADTrainer: {e}")
            logger.error("This usually means:")
            logger.error("1. You're not running from the project root directory")
            logger.error("2. The varibad package is not properly installed")
            logger.error("3. There are missing dependencies")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error(f"Python path: {sys.path}")
            
            # Try to give more specific help
            trainer_path = Path("varibad/core/trainer.py")
            if trainer_path.exists():
                logger.info(f"✅ Found trainer file at {trainer_path.absolute()}")
                logger.error("The file exists but can't be imported. Check for syntax errors or missing dependencies in the trainer module.")
            else:
                logger.error(f"❌ Trainer file not found at {trainer_path.absolute()}")
                logger.error("Make sure you're running from the correct directory")
            
            raise
        
        # Initialize trainer with validation
        logger.info("Initializing VariBAD trainer...")
        trainer = VariBADTrainer(
            data_path=data_path,
            episode_length=config.episode_length,
            action_dim=30,  # 30 S&P 500 companies
            latent_dim=config.latent_dim,
            max_episodes_buffer=config.buffer_size,
            enable_short_selling=config.short_selling,
            policy_lr=config.policy_lr,
            vae_encoder_lr=config.vae_encoder_lr,
            vae_decoder_lr=config.vae_decoder_lr,
            device=config.device
        )
        
        logger.info(f"✅ Trainer initialized successfully")
        logger.info(f"   State dimension: {trainer.state_dim}")
        logger.info(f"   Action dimension: {trainer.actual_action_dim}")
        logger.info(f"   Model parameters: {sum(p.numel() for p in trainer.varibad.parameters()):,}")
        
        # Train model
        logger.info("Starting training loop...")
        stats = trainer.train(
            num_iterations=config.num_iterations,
            episodes_per_iteration=config.episodes_per_iteration,
            vae_updates_per_iteration=config.vae_updates,
            eval_frequency=config.eval_frequency,
            save_frequency=config.save_frequency
        )
        
        # Save final model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = f"checkpoints/varibad_final_{timestamp}.pt"
        trainer.save_checkpoint(checkpoint_path)
        
        logger.info(f"✅ Training complete! Model saved to {checkpoint_path}")
        
        # Save training stats
        stats_path = f"results/training_stats_{timestamp}.json"
        try:
            import json
            with open(stats_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_stats = {}
                for key, value in stats.items():
                    if hasattr(value, 'tolist'):
                        json_stats[key] = value.tolist()
                    elif isinstance(value, (list, tuple)):
                        json_stats[key] = list(value)
                    else:
                        json_stats[key] = value
                json.dump(json_stats, f, indent=2)
            
            logger.info(f"📊 Training statistics saved to {stats_path}")
        except Exception as e:
            logger.warning(f"Failed to save training stats: {e}")
        
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
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint.get('config', {})
        
        # Recreate trainer
        trainer = VariBADTrainer(
            data_path=data_path,
            state_dim=config.get('state_dim'),
            action_dim=config.get('actual_action_dim', 60) // 2,
            device='cpu'
        )
        
        # Load model weights
        trainer.varibad.load_state_dict(checkpoint['varibad_state_dict'])
        
        # Run evaluation
        eval_stats = trainer.evaluate(num_episodes=50)
        
        logger.info("✅ Evaluation complete!")
        for key, value in eval_stats.items():
            logger.info(f"   {key}: {value:.4f}")
        
        # Save evaluation results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        eval_path = f"results/evaluation_{timestamp}.json"
        
        try:
            import json
            with open(eval_path, 'w') as f:
                json.dump(eval_stats, f, indent=2)
            logger.info(f"📊 Evaluation results saved to {eval_path}")
        except Exception as e:
            logger.warning(f"Failed to save evaluation results: {e}")
        
        return eval_stats
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def main():
    """Main pipeline entry point with comprehensive error handling."""
    parser = argparse.ArgumentParser(
        description="VariBAD Portfolio Optimization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python varibad/main.py --mode data_only                    # Process data only
  python varibad/main.py --mode train --num_iterations 100   # Quick training
  python varibad/main.py --mode evaluate --checkpoint path   # Evaluate model
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                      choices=['data_only', 'train', 'resume', 'evaluate'],
                      help='Pipeline mode to run')
    
    # Training parameters
    parser.add_argument('--num_iterations', type=int, default=1000,
                      help='Number of training iterations (default: 1000)')
    parser.add_argument('--episode_length', type=int, default=30,
                      help='Length of each trading episode (default: 30)')
    parser.add_argument('--episodes_per_iteration', type=int, default=5,
                      help='Episodes to collect per iteration (default: 5)')
    parser.add_argument('--vae_updates', type=int, default=10,
                      help='VAE updates per iteration (default: 10)')
    parser.add_argument('--latent_dim', type=int, default=5,
                      help='Latent dimension for VariBAD (default: 5)')
    parser.add_argument('--buffer_size', type=int, default=500,
                      help='Max episodes in trajectory buffer (default: 500)')
    
    # Model parameters
    parser.add_argument('--short_selling', action='store_true',
                      help='Enable short selling')
    parser.add_argument('--device', type=str, default='auto',
                      choices=['auto', 'cpu', 'cuda'],
                      help='Device to use (default: auto)'),
    parser.add_argument('--policy_lr', type=float, default=1e-4,
                        help='Learning rate for policy network (default: 1e-4)'),
    parser.add_argument('--vae_encoder_lr', type=float, default=1e-4,
                        help='Learning rate for VAE encoder network (default: 1e-4)'),
    parser.add_argument('--vae_decoder_lr', type=float, default=1e-4,
                        help='Learning rate for VAE decoder network (default: 1e-4)')
    
    # Evaluation parameters
    parser.add_argument('--eval_frequency', type=int, default=50,
                      help='Evaluate every N iterations (default: 50)')
    parser.add_argument('--save_frequency', type=int, default=100,
                      help='Save checkpoint every N iterations (default: 100)')
    
    # File paths
    parser.add_argument('--checkpoint', type=str,
                      help='Path to checkpoint for resume/evaluate modes')
    parser.add_argument('--data_path', type=str, default='data/sp500_rl_ready_cleaned.parquet',
                      help='Path to processed dataset')
    
    # Logging
    parser.add_argument('--log_level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.log_level)
    
    try:
        check_dependencies()
        create_directory_structure()
        
        # Auto-detect device
        if args.device == 'auto':
            try:
                import torch
                args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"Auto-detected device: {args.device}")
            except ImportError:
                args.device = 'cpu'
                logger.warning("PyTorch not found, using CPU")
        
        logger.info(f"🚀 Starting VariBAD pipeline in '{args.mode}' mode")
        logger.info(f"Configuration:")
        for key, value in vars(args).items():
            logger.info(f"  {key}: {value}")
        
        # Execute based on mode
        if args.mode == 'data_only':
            logger.info("📊 Data processing mode")
            final_data_path = download_and_clean_data(logger)
            logger.info(f"✅ Data processing complete! Dataset ready at: {final_data_path}")
        
        elif args.mode == 'train':
            logger.info("🏋️ Training mode")
            # Process data first if needed
            if not os.path.exists(args.data_path):
                logger.info("Data not found, processing first...")
                args.data_path = download_and_clean_data(logger)
            
            # Then train
            checkpoint_path, stats = train_varibad_model(args.data_path, args, logger)
            logger.info(f"✅ Training complete! Model saved to: {checkpoint_path}")
        
        elif args.mode == 'resume':
            logger.info("🔄 Resume training mode")
            if not args.checkpoint:
                logger.error("--checkpoint required for resume mode")
                logger.error("Example: python varibad/main.py --mode resume --checkpoint checkpoints/model.pt")
                sys.exit(1)
            
            # Ensure data exists
            if not os.path.exists(args.data_path):
                logger.info("Data not found, processing first...")
                args.data_path = download_and_clean_data(logger)
            
            logger.warning("Resume functionality not yet implemented")
            logger.info("Use the train mode with different parameters instead")
        
        elif args.mode == 'evaluate':
            logger.info("📊 Evaluation mode")
            if not args.checkpoint:
                logger.error("--checkpoint required for evaluate mode")
                logger.error("Example: python varibad/main.py --mode evaluate --checkpoint checkpoints/model.pt")
                sys.exit(1)
            
            # Ensure data exists
            if not os.path.exists(args.data_path):
                logger.info("Data not found, processing first...")
                args.data_path = download_and_clean_data(logger)
            
            eval_stats = evaluate_model(args.checkpoint, args.data_path, logger)
            logger.info("✅ Evaluation complete!")
    
    except KeyboardInterrupt:
        logger.info("⏹️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        logger.error("For debugging, try running with --log_level DEBUG")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    logger.info("🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()