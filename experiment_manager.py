import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import time
import psutil
import torch
import gc

from config import ExperimentConfig, experiment_to_training_config
#from mlflow_logger import MLflowIntegration


logger = logging.getLogger(__name__)

class ExperimentManager:
    """Manages batch experiment execution with checkpointing and recovery."""
    
    def __init__(self, 
                experiments: List[ExperimentConfig],
                checkpoint_dir: str = "experiment_checkpoints",
                max_retries: int = 0,
                force_recreate: bool = False):
        
        self.experiments = experiments
        self.max_retries = max_retries
        self.force_recreate = force_recreate
        
        # Create hierarchical checkpoint directory based on experiments
        if experiments:
            first_exp = experiments[0]
            exp_name = first_exp.exp_name or f"{first_exp.asset_class}_{first_exp.encoder}_study"
            encoder = first_exp.encoder
            asset_class = first_exp.asset_class
            self.checkpoint_dir = Path(exp_name) / encoder / asset_class / "experiment_checkpoints"
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self.results = {}
        self.failed_experiments = {}
        self.start_time = datetime.now()
        self.checkpoint_file = self.checkpoint_dir / "experiment_state.pkl"
        
        if self.force_recreate and self.checkpoint_file.exists():
            logger.info("⚡ Force recreate enabled: removing old checkpoint")
            self.checkpoint_file.unlink()

        # Load existing state if available
        self.load_checkpoint()
    
    def save_checkpoint(self):
        """Save current experiment state."""
        state = {
            'results': self.results,
            'failed_experiments': self.failed_experiments,
            'start_time': self.start_time,
            'completed_count': len(self.results),
            'failed_count': len(self.failed_experiments),
            'last_checkpoint': datetime.now()
        }
        
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(state, f)
    
    def load_checkpoint(self):
        """Load previous experiment state if exists."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    state = pickle.load(f)
                    
                self.results = state.get('results', {})
                self.failed_experiments = state.get('failed_experiments', {})
                self.start_time = state.get('start_time', datetime.now())
                
                completed = len(self.results)
                failed = len(self.failed_experiments)
                
                if completed > 0 or failed > 0:
                    logger.info(f"Resumed from checkpoint: {completed} completed, {failed} failed")
                    
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
    
    def get_pending_experiments(self) -> List[ExperimentConfig]:
        """Get list of experiments that haven't been completed or failed."""
        completed_names = set(self.results.keys())
        failed_names = set(self.failed_experiments.keys())
        processed_names = completed_names | failed_names
        
        pending = []
        for exp in self.experiments:
            exp_name = f"{exp.asset_class}_{exp.encoder}_seed{exp.seed}"
            if exp_name not in processed_names:
                pending.append(exp)
        
        return pending
    
    def run_single_experiment(self, exp_config: ExperimentConfig) -> Dict[str, Any]:
        """Run a single experiment with comprehensive error handling."""
        from main import run_training
        #import mlflow
        from csv_logger import CSVLogger
        
        cfg = experiment_to_training_config(exp_config)
        exp_name = cfg.exp_name
        
        #mlflow_integration = MLflowIntegration(run_name=cfg.exp_name, config=vars(cfg))
        #mlflow_integration.setup_mlflow()
        
        csv_logger = CSVLogger(run_name=cfg.exp_name, config=vars(cfg))
        csv_logger.setup_mlflow()
        
        logger.info(f"Starting experiment: {exp_name}")
        
        # System monitoring
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        try:
            # Log system info
            csv_logger.log_config()
            csv_logger.log_system_info(initial_memory)

            # Run training
            results = run_training(cfg)

            # Final system metrics
            final_memory = process.memory_info().rss / 1024 / 1024
            training_time = time.time() - start_time
            #mlflow_integration.log_final_system_metrics(final_memory, training_time, initial_memory)
            csv_logger.log_final_system_metrics(final_memory, training_time, initial_memory)
            
            # Success
            results['wall_time_seconds'] = training_time
            results['memory_peak_mb'] = final_memory
            results['experiment_name'] = exp_name
            results['success'] = True
            
            logger.info(f"✅ Completed: {exp_name}")
            return results
                
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM: {exp_name}")
            self._handle_cuda_oom()
            
            return {
                'success': False,
                'error_type': 'CUDA_OOM',
                'error_message': str(e)[:500],
                'experiment_name': exp_name,
                'wall_time_seconds': time.time() - start_time
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"Runtime OOM: {exp_name}")
                self._handle_cuda_oom()
                
                return {
                    'success': False,
                    'error_type': 'RUNTIME_OOM', 
                    'error_message': str(e)[:500],
                    'experiment_name': exp_name,
                    'wall_time_seconds': time.time() - start_time
                }
            else:
                raise
                
        except Exception as e:
            logger.error(f"Unexpected error in {exp_name}: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            
            return {
                'success': False,
                'error_type': type(e).__name__,
                'error_message': str(e)[:500],
                'experiment_name': exp_name,
                'wall_time_seconds': time.time() - start_time
            }
        
        finally:
            # Cleanup regardless of success/failure
            self._cleanup_experiment()
    
    def _handle_cuda_oom(self):
        """Handle CUDA out of memory errors."""
        logger.info("Handling CUDA OOM - cleaning up...")
        
        # Clear Python objects
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # Wait a moment for cleanup
        time.sleep(2)
        
        logger.info("CUDA cleanup completed")
    
    def _cleanup_experiment(self):
        """Clean up after each experiment."""
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Small delay to ensure cleanup
        time.sleep(1)
    
    def run_all_experiments(self, 
                          experiment_name: str = "portfolio_optimization_study",
                          save_progress_every: int = 5) -> Dict[str, Any]:
        """Run all pending experiments with progress tracking."""
        
        #import mlflow
        #mlflow.set_experiment(experiment_name)
        
        logger.info(f"Starting experiment batch: {experiment_name}")
        pending_experiments = self.get_pending_experiments()
        total_experiments = len(self.experiments)
        completed_at_start = len(self.results)
        
        
        logger.info(f"  Total experiments: {total_experiments}")
        logger.info(f"  Already completed: {completed_at_start}")  
        logger.info(f"  Already failed: {len(self.failed_experiments)}")
        logger.info(f"  Pending: {len(pending_experiments)}")
        
        if len(pending_experiments) == 0:
            logger.info("All experiments already completed!")
            return self.get_summary()
        
        # Process pending experiments
        for i, exp_config in enumerate(pending_experiments):
            exp_name = f"{exp_config.asset_class}_{exp_config.encoder}_seed{exp_config.seed}"
            current_number = completed_at_start + len(self.failed_experiments) + i + 1
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Experiment {current_number}/{total_experiments}: {exp_name}")
            logger.info(f"Asset: {exp_config.asset_class}, Encoder: {exp_config.encoder}, Seed: {exp_config.seed}")
            logger.info(f"{'='*60}")
            
            # Run experiment
            result = self.run_single_experiment(exp_config)
            
            # Store result
            if result['success']:
                self.results[exp_name] = result
            else:
                self.failed_experiments[exp_name] = result
            
            # Progress update
            completed_count = len(self.results)
            failed_count = len(self.failed_experiments)
            processed = completed_count + failed_count
            progress_pct = processed / total_experiments * 100
            
            logger.info(f"Progress: {progress_pct:.1f}% ({processed}/{total_experiments})")
            logger.info(f"Success rate: {completed_count}/{processed} ({completed_count/processed*100:.1f}%)")
            
            # Save checkpoint periodically
            if (i + 1) % save_progress_every == 0:
                self.save_checkpoint()
                logger.info(f"Checkpoint saved after {processed} experiments")
        
        # Final checkpoint
        self.save_checkpoint()
        
        # Generate summary
        summary = self.get_summary()
        self.save_final_report()
        
        return summary
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate experiment batch summary."""
        total = len(self.experiments)
        completed = len(self.results)
        failed = len(self.failed_experiments)
        
        # Success metrics
        success_rate = completed / total if total > 0 else 0
        
        # Timing
        elapsed_time = datetime.now() - self.start_time
        avg_time_per_exp = elapsed_time.total_seconds() / max(completed + failed, 1)
        
        # Error analysis
        error_types = {}
        for result in self.failed_experiments.values():
            error_type = result.get('error_type', 'Unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Performance stats from successful experiments
        if completed > 0:
            test_rewards = [r.get('final_test_reward', 0) for r in self.results.values() if r.get('final_test_reward')]
            sharpe_ratios = [r.get('backtest_sharpe', 0) for r in self.results.values() if r.get('backtest_sharpe')]
            
            performance_stats = {
                'avg_test_reward': sum(test_rewards) / len(test_rewards) if test_rewards else 0,
                'best_test_reward': max(test_rewards) if test_rewards else 0,
                'avg_sharpe': sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0,
                'best_sharpe': max(sharpe_ratios) if sharpe_ratios else 0
            }
        else:
            performance_stats = {}
        
        # Resource usage summary (from completed experiments)
        if completed > 0:
            memory_values = [r.get('memory_peak_mb', 0) for r in self.results.values()]
            time_values = [r.get('wall_time_seconds', 0) for r in self.results.values()]
            
            resource_stats = {
                'avg_memory_mb': sum(memory_values) / len(memory_values) if memory_values else 0,
                'max_memory_mb': max(memory_values) if memory_values else 0,
                'avg_time_minutes': (sum(time_values) / len(time_values) / 60) if time_values else 0,
                'total_time_hours': sum(time_values) / 3600 if time_values else 0
            }
        else:
            resource_stats = {}
        
        summary = {
            'batch_info': {
                'total_experiments': total,
                'completed': completed,
                'failed': failed,
                'success_rate': success_rate,
                'elapsed_time_hours': elapsed_time.total_seconds() / 3600,
                'avg_time_per_experiment_minutes': avg_time_per_exp / 60
            },
            'error_analysis': error_types,
            'performance_stats': performance_stats,
            'resource_usage': resource_stats,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat()
        }
        
        return summary
    
    def save_final_report(self):
        """Save comprehensive final report."""
        summary = self.get_summary()
        
        # Save summary JSON
        summary_file = self.checkpoint_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        results_file = self.checkpoint_dir / "detailed_results.json"
        all_results = {
            'successful_experiments': self.results,
            'failed_experiments': self.failed_experiments,
            'summary': summary
        }
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Final report saved to {self.checkpoint_dir}")
        
        # Print final summary
        self.print_final_summary(summary)
    
    def print_final_summary(self, summary: Dict[str, Any]):
        """Print human-readable final summary."""
        batch_info = summary['batch_info']
        
        print(f"\n{'='*80}")
        print("EXPERIMENT BATCH COMPLETE")
        print(f"{'='*80}")
        print(f"Total experiments: {batch_info['total_experiments']}")
        print(f"Successful: {batch_info['completed']} ({batch_info['success_rate']*100:.1f}%)")
        print(f"Failed: {batch_info['failed']}")
        print(f"Total time: {batch_info['elapsed_time_hours']:.1f} hours")
        print(f"Average time per experiment: {batch_info['avg_time_per_experiment_minutes']:.1f} minutes")
        
        if summary['error_analysis']:
            print(f"\nError breakdown:")
            for error_type, count in summary['error_analysis'].items():
                print(f"  {error_type}: {count}")
        
        if summary['performance_stats']:
            perf = summary['performance_stats']
            print(f"\nPerformance summary:")
            print(f"  Average test reward: {perf.get('avg_test_reward', 0):.4f}")
            print(f"  Best test reward: {perf.get('best_test_reward', 0):.4f}")
            print(f"  Average Sharpe ratio: {perf.get('avg_sharpe', 0):.4f}")
            print(f"  Best Sharpe ratio: {perf.get('best_sharpe', 0):.4f}")
        
        if summary['resource_usage']:
            res = summary['resource_usage']
            print(f"\nResource usage summary:")
            print(f"  Average memory: {res.get('avg_memory_mb', 0):.1f}MB")
            print(f"  Peak memory: {res.get('max_memory_mb', 0):.1f}MB")
            print(f"  Average experiment time: {res.get('avg_time_minutes', 0):.1f} minutes")
            print(f"  Total compute time: {res.get('total_time_hours', 0):.1f} hours")
        
        print(f"{'='*80}")