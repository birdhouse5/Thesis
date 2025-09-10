#!/usr/bin/env python3
"""
Simple script to run DSR hyperparameter optimization studies.

This script will:
1. Run HPO for SP500 with 5 trials
2. Run HPO for Crypto with 5 trials  
3. Generate a summary report with optimal parameters

Usage:
    python run_dsr_hpo.py
"""

import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run DSR HPO studies for both asset classes."""
    
    logger.info("Starting DSR Hyperparameter Optimization")
    
    # Import here to ensure logging is set up
    from dsr_hpo_study import run_asset_class_study, summarize_all_studies
    from mlflow_setup import setup_mlflow
    import mlflow
    
    # Set up MLflow
    if not setup_mlflow():
        logger.warning("MLflow setup failed - proceeding with local tracking")
        mlflow.set_tracking_uri("file:./mlruns")
    
    mlflow.set_experiment("dsr_hyperparameter_optimization")
    
    # Run studies for both asset classes
    n_trials = 5
    encoder = "vae"  # Focus on VAE encoder for now
    
    asset_classes = ["sp500", "crypto"]
    
    for asset_class in asset_classes:
        logger.info(f"\n{'-'*50}")
        logger.info(f"Running DSR HPO for {asset_class.upper()}")
        logger.info(f"Trials: {n_trials}, Encoder: {encoder}")
        logger.info(f"{'-'*50}")
        
        try:
            study = run_asset_class_study(asset_class, n_trials, encoder)
            logger.info(f"✅ Completed HPO for {asset_class}")
            logger.info(f"   Best value: {study.best_value:.6f}")
            logger.info(f"   Best params: {study.best_params}")
            
        except Exception as e:
            logger.error(f"❌ HPO failed for {asset_class}: {str(e)}")
            continue
    
    # Generate summary report
    logger.info(f"\n{'-'*50}")
    logger.info("Generating summary report...")
    logger.info(f"{'-'*50}")
    
    summarize_all_studies()
    
    # Instructions for next steps
    print(f"\n{'='*60}")
    print("DSR HYPERPARAMETER OPTIMIZATION COMPLETE!")
    print(f"{'='*60}")
    print("Next steps:")
    print("1. Review results in dsr_hpo_results/ directory")
    print("2. Check dsr_hpo_combined_summary.json for optimal parameters")
    print("3. Update your config.py with the optimal DSR parameters")
    print("4. Run full experiments with tuned parameters")
    print(f"{'='*60}")
    
    # Show where results are saved
    results_dir = Path("dsr_hpo_results")
    if results_dir.exists():
        files = list(results_dir.glob("*"))
        print(f"\nResults saved in {results_dir}:")
        for file in sorted(files):
            print(f"  - {file.name}")
    
    logger.info("DSR HPO complete!")


if __name__ == "__main__":
    main()