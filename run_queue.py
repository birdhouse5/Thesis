# run_queue.py
import subprocess
import json
import shutil
import time
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import glob
import sys

class ExperimentQueue:
    def __init__(self, config_dir="experiments/configs", results_dir="experiments/results"):
        self.config_dir = Path(config_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.status_file = self.results_dir / "queue_status.json"
        self.summary_file = self.results_dir / "experiment_summary.csv"
        
        # Load or initialize status
        self.status = self.load_status()
        
    def load_status(self):
        """Load queue status from file"""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                return json.load(f)
        
        return {
            "completed": [],
            "failed": [],
            "current": None,
            "queue_start_time": None,
            "total_experiments": 0
        }
    
    def save_status(self):
        """Save current status to file"""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2, default=str)
    
    def discover_configs(self):
        """Find all config files in config directory"""
        config_files = list(self.config_dir.glob("*.json"))
        config_files.sort()  # Consistent ordering
        
        print(f"Found {len(config_files)} config files in {self.config_dir}")
        return config_files
    
    def get_remaining_configs(self, all_configs):
        """Get configs that haven't been completed yet"""
        completed_names = {exp["config_file"] for exp in self.status["completed"]}
        failed_names = {exp["config_file"] for exp in self.status["failed"]}
        
        remaining = []
        for config_file in all_configs:
            if config_file.name not in completed_names and config_file.name not in failed_names:
                remaining.append(config_file)
        
        return remaining
    
    def run_experiment(self, config_file):
        """Run a single experiment and collect results"""
        print(f"\n{'='*80}")
        print(f"RUNNING: {config_file.name}")
        print(f"{'='*80}")
        
        start_time = datetime.now()
        
        # Load config to get experiment name
        with open(config_file, 'r') as f:
            config = json.load(f)
        exp_name = config.get("exp_name", config_file.stem)
        
        # Update status
        self.status["current"] = {
            "config_file": config_file.name,
            "exp_name": exp_name,
            "start_time": start_time.isoformat()
        }
        self.save_status()
        
        # Create experiment result directory
        exp_result_dir = self.results_dir / exp_name
        exp_result_dir.mkdir(exist_ok=True)
        
        # Copy config file to results
        shutil.copy2(config_file, exp_result_dir / "config.json")
        
        # Prepare command
        cmd = [
            sys.executable, "main.py", 
            "--config", str(config_file)
        ]
        
        # Prepare log files
        stdout_log = exp_result_dir / "console.log"
        stderr_log = exp_result_dir / "error.log"
        
        success = False
        error_msg = None
        
        try:
            print(f"Command: {' '.join(cmd)}")
            print(f"Logs: {stdout_log}")
            
            # Run experiment
            with open(stdout_log, 'w') as out, open(stderr_log, 'w') as err:
                process = subprocess.run(
                    cmd,
                    stdout=out,
                    stderr=err,
                    text=True,
                    timeout=7200  # 2 hour timeout
                )
            
            success = (process.returncode == 0)
            if not success:
                with open(stderr_log, 'r') as f:
                    error_msg = f.read()
        
        except subprocess.TimeoutExpired:
            error_msg = "Experiment timed out after 2 hours"
        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Collect results regardless of success/failure
        result_summary = self.collect_experiment_results(
            exp_result_dir, config, success, error_msg, duration
        )
        
        # Update status
        if success:
            self.status["completed"].append({
                "config_file": config_file.name,
                "exp_name": exp_name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_minutes": duration.total_seconds() / 60,
                "result": result_summary
            })
            print(f"✅ SUCCESS: {exp_name} ({duration})")
        else:
            self.status["failed"].append({
                "config_file": config_file.name,
                "exp_name": exp_name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_minutes": duration.total_seconds() / 60,
                "error": error_msg,
                "result": result_summary
            })
            print(f"❌ FAILED: {exp_name} - {error_msg}")
        
        self.status["current"] = None
        self.save_status()
        
        return success
    
    def collect_experiment_results(self, exp_result_dir, config, success, error_msg, duration):
        """Collect and organize all experiment outputs"""
        print(f"📦 Collecting results to {exp_result_dir}")
        
        result_summary = {
            "success": success,
            "error": error_msg,
            "duration_minutes": duration.total_seconds() / 60
        }
        
        try:
            # Look for logs directory (created by main.py)
            logs_pattern = "logs/" + config.get("exp_name", "*") + "_*"
            log_dirs = glob.glob(logs_pattern)
            
            if log_dirs:
                # Get the most recent log directory
                latest_log_dir = max(log_dirs, key=lambda x: Path(x).stat().st_mtime)
                latest_log_path = Path(latest_log_dir)
                
                # First, copy and parse CSV files BEFORE moving
                csv_source = latest_log_path / "summary"
                if csv_source.exists():
                    csv_target = exp_result_dir / "csvs"
                    print(f"  📈 Copying CSV files: {csv_source} -> {csv_target}")
                    shutil.copytree(csv_source, csv_target)
                    
                    # Parse final results from CSV
                    result_summary.update(self.parse_csv_results(csv_target))
                
                # Then move the entire log directory to tensorboard
                target_tb_dir = exp_result_dir / "tensorboard"
                print(f"  📊 Moving TensorBoard logs: {latest_log_dir} -> {target_tb_dir}")
                if latest_log_path.exists():
                    shutil.move(latest_log_dir, target_tb_dir)
            
            # Look for checkpoint files
            checkpoint_pattern = "checkpoints/*"
            checkpoint_files = glob.glob(checkpoint_pattern)
            
            if checkpoint_files:
                checkpoint_dir = exp_result_dir / "checkpoints"
                checkpoint_dir.mkdir(exist_ok=True)
                
                for checkpoint_file in checkpoint_files:
                    if config.get("exp_name", "") in checkpoint_file:
                        target_file = checkpoint_dir / Path(checkpoint_file).name
                        print(f"  💾 Moving checkpoint: {checkpoint_file} -> {target_file}")
                        shutil.move(checkpoint_file, target_file)
            
            # Create experiment summary
            summary = {
                "config": config,
                "success": success,
                "error": error_msg,
                "duration_minutes": duration.total_seconds() / 60,
                "results": result_summary,
                "collected_at": datetime.now().isoformat()
            }
            
            with open(exp_result_dir / "summary.json", 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
        except Exception as e:
            print(f"  ⚠️  Error collecting results: {e}")
            result_summary["collection_error"] = str(e)
        
        return result_summary
    
    def parse_csv_results(self, csv_dir):
        """Extract key metrics from CSV files"""
        results = {}
        
        try:
            # Look for test results
            test_csv = csv_dir / "test_metrics.csv"
            if test_csv.exists():
                import pandas as pd
                df = pd.read_csv(test_csv)
                if not df.empty:
                    results["final_test_reward"] = float(df["avg_reward"].iloc[-1])
                    results["final_test_return"] = float(df["avg_return"].iloc[-1])
            
            # Look for training results
            train_csv = csv_dir / "training_metrics.csv"
            if train_csv.exists():
                import pandas as pd
                df = pd.read_csv(train_csv)
                if not df.empty:
                    results["final_training_reward"] = float(df["episode_reward"].iloc[-1])
                    results["avg_training_reward"] = float(df["episode_reward"].mean())
            
            # Look for episode details
            episodes_csv = csv_dir / "episode_details.csv"
            if episodes_csv.exists():
                import pandas as pd
                df = pd.read_csv(episodes_csv)
                if not df.empty:
                    results["total_episodes"] = len(df)
                    results["avg_sharpe_ratio"] = float(df["sharpe_ratio"].mean())
                    results["final_cumulative_return"] = float(df["cumulative_return"].iloc[-1])
        
        except Exception as e:
            results["csv_parse_error"] = str(e)
        
        return results
    
    def print_progress(self, current_idx, total_experiments):
        """Print current progress and ETA"""
        completed = len(self.status["completed"])
        failed = len(self.status["failed"])
        remaining = total_experiments - completed - failed
        
        print(f"\n📊 PROGRESS: {completed + failed}/{total_experiments} experiments completed")
        print(f"   ✅ Successful: {completed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏳ Remaining: {remaining}")
        
        # Calculate ETA
        if completed > 0 and self.status.get("queue_start_time"):
            start_time = datetime.fromisoformat(self.status["queue_start_time"])
            elapsed = datetime.now() - start_time
            avg_time_per_exp = elapsed / (completed + failed)
            eta = datetime.now() + (avg_time_per_exp * remaining)
            
            print(f"   🕐 Elapsed: {elapsed}")
            print(f"   ⏰ ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def generate_summary_report(self):
        """Generate final summary report"""
        print(f"\n📋 Generating summary report...")
        
        # Create CSV summary
        import pandas as pd
        
        all_experiments = self.status["completed"] + self.status["failed"]
        if not all_experiments:
            return
        
        # Flatten data for CSV
        csv_data = []
        for exp in all_experiments:
            row = {
                "config_file": exp["config_file"],
                "exp_name": exp["exp_name"],
                "success": exp.get("result", {}).get("success", False),
                "duration_minutes": exp["duration_minutes"],
                "final_test_reward": exp.get("result", {}).get("final_test_reward", None),
                "final_test_return": exp.get("result", {}).get("final_test_return", None),
                "total_episodes": exp.get("result", {}).get("total_episodes", None),
            }
            
            # Add config parameters
            config_file = self.config_dir / exp["config_file"]
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                row.update({
                    "latent_dim": config.get("latent_dim"),
                    "hidden_dim": config.get("hidden_dim"),
                    "vae_lr": config.get("vae_lr"),
                    "policy_lr": config.get("policy_lr"),
                    "batch_size": config.get("batch_size"),
                    "vae_beta": config.get("vae_beta"),
                })
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(self.summary_file, index=False)
        
        # Print top performers
        successful_df = df[df["success"] == True]
        if not successful_df.empty and "final_test_reward" in successful_df.columns:
            # Filter out None values and convert to numeric
            valid_rewards = successful_df.dropna(subset=["final_test_reward"])
            if not valid_rewards.empty:
                valid_rewards["final_test_reward"] = pd.to_numeric(valid_rewards["final_test_reward"], errors='coerce')
                valid_rewards = valid_rewards.dropna(subset=["final_test_reward"])
                
                if not valid_rewards.empty:
                    print(f"\nTOP 5 PERFORMERS:")
                    top_5 = valid_rewards.nlargest(5, "final_test_reward")
                    for idx, row in top_5.iterrows():
                        print(f"  {row['exp_name']}: {row['final_test_reward']:.4f}")
                else:
                    print(f"\nNo experiments with valid test rewards found")
            else:
                print(f"\nNo experiments with test rewards found")
        else:
            print(f"\nNo successful experiments found")
        
        print(f"\n📄 Summary saved to: {self.summary_file}")
    
    def run_queue(self, resume=False, max_experiments=None):
        """Run the entire experiment queue"""
        all_configs = self.discover_configs()
        
        if not all_configs:
            print("No config files found!")
            return
        
        if resume:
            remaining_configs = self.get_remaining_configs(all_configs)
            print(f"Resuming: {len(remaining_configs)} experiments remaining")
        else:
            remaining_configs = all_configs
            # Reset status for fresh run
            self.status = {
                "completed": [],
                "failed": [],
                "current": None,
                "queue_start_time": datetime.now().isoformat(),
                "total_experiments": len(remaining_configs)
            }
        
        if max_experiments:
            remaining_configs = remaining_configs[:max_experiments]
        
        total_experiments = len(remaining_configs)
        self.status["total_experiments"] = total_experiments
        self.save_status()
        
        print(f"🚀 Starting queue: {total_experiments} experiments")
        queue_start = datetime.now()
        
        try:
            for i, config_file in enumerate(remaining_configs):
                self.print_progress(i, total_experiments)
                
                success = self.run_experiment(config_file)
                
                # Brief pause between experiments
                time.sleep(2)
        
        except KeyboardInterrupt:
            print(f"\n🛑 Queue interrupted by user")
            print(f"To resume: python run_queue.py --resume")
        
        except Exception as e:
            print(f"\n💥 Queue failed with error: {e}")
            raise
        
        finally:
            # Generate final report
            queue_end = datetime.now()
            total_duration = queue_end - queue_start
            
            print(f"\n🏁 QUEUE COMPLETE")
            print(f"Total time: {total_duration}")
            print(f"Completed: {len(self.status['completed'])}")
            print(f"Failed: {len(self.status['failed'])}")
            
            self.generate_summary_report()


def main():
    parser = argparse.ArgumentParser(description="Run experiment queue")
    parser.add_argument("--config_dir", type=str, default="experiments/configs", 
                       help="Directory containing config files")
    parser.add_argument("--results_dir", type=str, default="experiments/results",
                       help="Directory to store results")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume from previous run")
    parser.add_argument("--max_experiments", type=int,
                       help="Limit number of experiments to run")
    
    args = parser.parse_args()
    
    # Create queue runner
    queue = ExperimentQueue(args.config_dir, args.results_dir)
    
    # Run the queue
    queue.run_queue(resume=args.resume, max_experiments=args.max_experiments)


if __name__ == "__main__":
    main()