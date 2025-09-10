#!/usr/bin/env python3
"""
DSR Hyperparameter Optimization Study

This script runs Optuna studies to find optimal DSR parameters for each asset class.
It runs shortened training (1000 episodes) to identify good parameter values that 
can then be used in the full experiment pipeline.

Usage:
    python dsr_hpo_study.py --asset_class sp500 --n_trials 5
    python dsr_hpo_study.py --asset_class crypto --n_trials 5
    python dsr_hpo_study.py --asset_class all --n_trials 5  # Run both
"""

import argparse
import logging
import os
import json
import mlflow
import optuna
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Import your modules
from config import create_hpo_config
from main import run_training, setup_debug_logging, cleanup_gpu_memory
from mlflow_setup import setup_mlflow

# Set up logging
logger = logging.getLogger(__name__)

class DSROptimizer:
    """Manages DSR hyperparameter optimization for a specific asset class."""
    
    def __init__(self, asset_class: str, n_trials: int = 5, encoder: str = "vae"):
        self.asset_class = asset_class
        self.n_trials = n_trials
        self.encoder = encoder
        self.results_dir = Path("dsr_hpo_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Study configuration
        self.study_name = f"dsr_hpo_{asset_class}_{encoder}"
        self.storage_file = self.results_dir / f"{self.study_name}.db"
        
        logger.info(f"Initialized DSR optimizer for {asset_class}")
        logger.info(f"Study: {self.study_name}, Trials: {n_trials}")
    
    def objective(self, trial):
        """
        Optuna objective function for DSR parameter optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation reward (higher is better)
        """
        # Suggest DSR parameters
        eta = trial.suggest_float('eta', 0.001, 0.3, log=True)
        rf_rate = 0.02
        transaction_cost_rate = 0.0001
        
        # Create HPO config with suggested parameters
        try:
            cfg = create_hpo_config(
                asset_class=self.asset_class,
                encoder=self.encoder,
                seed=42,  # Fixed seed for HPO reproducibility
                eta=eta,
                rf_rate=rf_rate,
                transaction_cost_rate=transaction_cost_rate
            )
            
            logger.info(f"Trial {trial.number}: eta={eta:.4f}, rf_rate={rf_rate:.4f}, tx_cost={transaction_cost_rate:.4f}")
            
            # Run training with MLflow tracking
            with mlflow.start_run(run_name=f"{self.study_name}_trial_{trial.number}"):
                # Log trial parameters
                mlflow.log_param("trial_number", trial.number)
                mlflow.log_param("eta", eta)
                mlflow.log_param("rf_rate", rf_rate)
                mlflow.log_param("transaction_cost_rate", transaction_cost_rate)
                mlflow.log_param("asset_class", self.asset_class)
                mlflow.log_param("encoder", self.encoder)
                mlflow.log_param("study_name", self.study_name)
                
                # Run training
                results = run_training(cfg)
                
                # Extract objective value
                if results.get("training_completed", False):
                    # Use best validation reward as objective
                    objective_value = results.get("best_val_reward", float('-inf'))
                    
                    # Log additional metrics
                    mlflow.log_metric("objective_value", objective_value)
                    mlflow.log_metric("final_test_reward", results.get("final_test_reward", 0))
                    mlflow.log_metric("backtest_sharpe", results.get("backtest_sharpe", 0))
                    mlflow.log_metric("episodes_trained", results.get("episodes_trained", 0))
                    
                    logger.info(f"Trial {trial.number} completed: objective={objective_value:.4f}")
                    
                else:
                    # Training failed
                    objective_value = float('-inf')
                    mlflow.log_metric("objective_value", objective_value)
                    logger.warning(f"Trial {trial.number} failed: {results.get('error', 'Unknown error')}")
                
                return objective_value
                
        except Exception as e:
            logger.error(f"Trial {trial.number} crashed: {str(e)}")
            
            # Log error to MLflow
            try:
                mlflow.log_param("error", str(e))
                mlflow.log_metric("objective_value", float('-inf'))
            except:
                pass
            
            # Return very poor objective
            return float('-inf')
        
        finally:
            # Cleanup after each trial
            cleanup_gpu_memory()
    
    def run_study(self):
        """Run the complete optimization study."""
        
        logger.info(f"Starting DSR optimization study for {self.asset_class}")
        
        # Create or load study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=f"sqlite:///{self.storage_file}",
            direction="maximize",
            load_if_exists=True
        )
        
        # Get initial trial count
        initial_trials = len(study.trials)
        remaining_trials = max(0, self.n_trials - initial_trials)
        
        if remaining_trials == 0:
            logger.info(f"Study already completed with {initial_trials} trials")
        else:
            logger.info(f"Running {remaining_trials} additional trials (total target: {self.n_trials})")
            
            # Run optimization
            study.optimize(self.objective, n_trials=remaining_trials)
        
        # Save and report results
        self.save_results(study)
        self.print_results(study)
        
        return study
    
    def save_results(self, study):
        """Save study results to files."""
        
        # Study summary
        results_summary = {
            "study_name": self.study_name,
            "asset_class": self.asset_class,
            "encoder": self.encoder,
            "n_trials": len(study.trials),
            "best_value": study.best_value,
            "best_params": study.best_params,
            "completed_at": datetime.now().isoformat()
        }
        
        # Save summary
        summary_file = self.results_dir / f"{self.study_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Save detailed trial results
        trials_data = []
        for trial in study.trials:
            trial_data = {
                "trial_number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
                "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None
            }
            trials_data.append(trial_data)
        
        trials_file = self.results_dir / f"{self.study_name}_trials.json"
        with open(trials_file, 'w') as f:
            json.dump(trials_data, f, indent=2)
        
        # Save as CSV for easy analysis
        df = study.trials_dataframe()
        if not df.empty:
            csv_file = self.results_dir / f"{self.study_name}_trials.csv"
            df.to_csv(csv_file, index=False)
        
        logger.info(f"Results saved to {self.results_dir}")
    
    def print_results(self, study):
        """Print study results summary."""
        
        print(f"\n{'='*60}")
        print(f"DSR HPO STUDY RESULTS: {self.asset_class.upper()}")
        print(f"{'='*60}")
        print(f"Study name: {self.study_name}")
        print(f"Total trials: {len(study.trials)}")
        print(f"Best value: {study.best_value:.6f}")
        print(f"Best parameters:")
        for param, value in study.best_params.items():
            print(f"  {param}: {value:.6f}")
        
        # Show top 3 trials
        print(f"\nTop 3 trials:")
        sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)
        for i, trial in enumerate(sorted_trials[:3]):
            if trial.value is not None:
                print(f"  {i+1}. Trial {trial.number}: {trial.value:.6f}")
                for param, value in trial.params.items():
                    print(f"     {param}: {value:.6f}")
        
        print(f"{'='*60}")


def run_asset_class_study(asset_class: str, n_trials: int = 5, encoder: str = "vae"):
    """Run DSR optimization for a specific asset class."""
    
    optimizer = DSROptimizer(asset_class, n_trials, encoder)
    study = optimizer.run_study()
    return study


def summarize_all_studies(results_dir: Path = None):
    """Create a summary report of all DSR studies."""
    
    if results_dir is None:
        results_dir = Path("dsr_hpo_results")
    
    if not results_dir.exists():
        logger.warning("No HPO results directory found")
        return
    
    # Find all summary files
    summary_files = list(results_dir.glob("*_summary.json"))
    
    if not summary_files:
        logger.warning("No study summaries found")
        return
    
    # Load and combine summaries
    all_summaries = []
    for summary_file in summary_files:
        with open(summary_file, 'r') as f:
            summary = json.load(f)
            all_summaries.append(summary)
    
    # Create combined report
    print(f"\n{'='*80}")
    print("DSR HYPERPARAMETER OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    
    for summary in sorted(all_summaries, key=lambda x: x['asset_class']):
        print(f"\nAsset Class: {summary['asset_class'].upper()}")
        print(f"  Study: {summary['study_name']}")
        print(f"  Trials: {summary['n_trials']}")
        print(f"  Best value: {summary['best_value']:.6f}")
        print(f"  Best parameters:")
        for param, value in summary['best_params'].items():
            print(f"    {param}: {value:.6f}")
    
    # Save combined summary
    combined_file = results_dir / "dsr_hpo_combined_summary.json"
    with open(combined_file, 'w') as f:
        json.dump({
            "summaries": all_summaries,
            "generated_at": datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nCombined summary saved to: {combined_file}")
    print(f"{'='*80}")


def main():
    """Main HPO script."""
    
    parser = argparse.ArgumentParser(description="DSR Hyperparameter Optimization")
    parser.add_argument("--asset_class", choices=["sp500", "crypto", "all"], 
                       default="all", help="Asset class to optimize")
    parser.add_argument("--n_trials", type=int, default=5, 
                       help="Number of optimization trials")
    parser.add_argument("--encoder", choices=["vae", "none", "hmm"], 
                       default="vae", help="Encoder type")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set up environment
    if args.debug:
        os.environ["DEBUG"] = "true"
    
    setup_debug_logging()
    
    # Set up MLflow
    if not setup_mlflow():
        logger.error("Failed to setup MLflow - cannot proceed")
        return
    
    mlflow.set_experiment("dsr_hyperparameter_optimization")
    
    logger.info("Starting DSR hyperparameter optimization")
    logger.info(f"Asset class: {args.asset_class}")
    logger.info(f"Trials per study: {args.n_trials}")
    logger.info(f"Encoder: {args.encoder}")
    
    # Run optimization studies
    if args.asset_class == "all":
        # Run for both asset classes
        for asset_class in ["sp500", "crypto"]:
            logger.info(f"\n{'-'*40}")
            logger.info(f"Starting study for {asset_class}")
            logger.info(f"{'-'*40}")
            
            try:
                run_asset_class_study(asset_class, args.n_trials, args.encoder)
            except Exception as e:
                logger.error(f"Study failed for {asset_class}: {str(e)}")
                continue
    else:
        # Run for specific asset class
        run_asset_class_study(args.asset_class, args.n_trials, args.encoder)
    
    # Generate summary report
    logger.info("\nGenerating summary report...")
    summarize_all_studies()
    
    logger.info("DSR hyperparameter optimization complete!")


if __name__ == "__main__":
    main()