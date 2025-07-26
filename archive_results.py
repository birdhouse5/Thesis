#!/usr/bin/env python3
"""
VariBAD Training Results Archival System
Creates self-contained training result packages for future reference
"""

import os
import json
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import glob
import zipfile
from datetime import datetime
import platform
import subprocess
import sys
from pathlib import Path

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

class VariBADResultsArchiver:
    """
    Complete training results archival system that creates self-contained
    packages with all context needed for future interpretation.
    """
    
    def __init__(self, run_name=None):
        """
        Initialize archiver.
        
        Args:
            run_name: Optional custom name for this training run
        """
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = run_name or f"varibad_training_{self.timestamp}"
        self.archive_dir = Path(f"archives/{self.run_name}")
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📦 Creating training archive: {self.archive_dir}")
    
    def capture_system_info(self):
        """Capture complete system and environment information."""
        print("🖥️  Capturing system information...")
        
        system_info = {
            "timestamp": self.timestamp,
            "run_name": self.run_name,
            "system": {
                "platform": platform.platform(),
                "python_version": sys.version,
                "hostname": platform.node(),
                "cpu_count": os.cpu_count(),
                "cwd": os.getcwd()
            },
            "environment": {
                "conda_env": os.environ.get('CONDA_DEFAULT_ENV', 'None'),
                "virtual_env": os.environ.get('VIRTUAL_ENV', 'None'),
                "python_path": sys.executable
            }
        }
        
        # GPU Information
        try:
            import torch
            system_info["gpu"] = {
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
            }
        except ImportError:
            system_info["gpu"] = {"error": "PyTorch not available"}
        
        # Package versions
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], 
                                  capture_output=True, text=True)
            system_info["packages"] = result.stdout.strip().split('\n')
        except:
            system_info["packages"] = ["Could not capture package list"]
        
        # Git information (if available)
        try:
            git_hash = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                    capture_output=True, text=True).stdout.strip()
            git_branch = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                                      capture_output=True, text=True).stdout.strip()
            system_info["git"] = {
                "commit_hash": git_hash,
                "branch": git_branch,
                "has_uncommitted_changes": len(subprocess.run(['git', 'diff', '--name-only'], 
                                                            capture_output=True, text=True).stdout.strip()) > 0
            }
        except:
            system_info["git"] = {"error": "Git information not available"}
        
        # Save system info
        with open(self.archive_dir / "system_info.json", 'w') as f:
            json.dump(system_info, f, indent=2)
        
        return system_info
    
    def capture_training_config(self):
        """Capture complete training configuration."""
        print("⚙️  Capturing training configuration...")
        
        config = {
            "data_source": "S&P 500 companies (1990-2025)",
            "assets": ["IBM", "MSFT", "ORCL", "INTC", "HPQ", "CSCO", "JPM", "BAC", "WFC", "C", "AXP",
                      "JNJ", "PFE", "MRK", "ABT", "KO", "PG", "WMT", "PEP", "XOM", "CVX", "COP",
                      "GE", "CAT", "BA", "HD", "MCD", "SO", "D", "DD"],
            "algorithm": "VariBAD (Variational Bayes Adaptive Deep RL)",
            "paper_reference": "Zintgraf et al. 2020 - VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning"
        }
        
        # Try to extract actual config from checkpoints
        checkpoints = list(Path("checkpoints").glob("*.pt")) if Path("checkpoints").exists() else []
        if checkpoints:
            try:
                latest_checkpoint = max(checkpoints, key=os.path.getctime)
                checkpoint_data = torch.load(latest_checkpoint, map_location='cpu')
                if 'config' in checkpoint_data:
                    config["model_config"] = checkpoint_data['config']
            except Exception as e:
                config["model_config_error"] = str(e)
        
        # Try to extract training parameters from logs
        log_files = list(Path("logs").glob("*.log")) if Path("logs").exists() else []
        if log_files:
            latest_log = max(log_files, key=os.path.getctime)
            try:
                with open(latest_log, 'r') as f:
                    log_content = f.read()
                
                # Extract key parameters from log
                import re
                params = {}
                
                # Look for parameter patterns
                param_patterns = {
                    'num_iterations': r'num_iterations[:\s]+(\d+)',
                    'episode_length': r'episode_length[:\s]+(\d+)',
                    'latent_dim': r'latent_dim[:\s]+(\d+)',
                    'device': r'device[:\s]+(\w+)',
                    'short_selling': r'short_selling[:\s]+(\w+)',
                }
                
                for param, pattern in param_patterns.items():
                    match = re.search(pattern, log_content, re.IGNORECASE)
                    if match:
                        params[param] = match.group(1)
                
                config["training_parameters"] = params
                
            except Exception as e:
                config["log_parsing_error"] = str(e)
        
        # Save config
        with open(self.archive_dir / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        return config
    
    def capture_data_info(self):
        """Capture information about the training dataset."""
        print("📊 Capturing dataset information...")
        
        data_info = {}
        
        # Look for data files
        data_files = list(Path("data").glob("*.parquet")) if Path("data").exists() else []
        data_files.extend(list(Path("data").glob("*.csv")) if Path("data").exists() else [])
        
        if data_files:
            try:
                # Use the cleaned dataset if available
                cleaned_data_file = None
                for f in data_files:
                    if 'cleaned' in f.name:
                        cleaned_data_file = f
                        break
                
                if not cleaned_data_file and data_files:
                    cleaned_data_file = data_files[0]
                
                if cleaned_data_file:
                    if cleaned_data_file.suffix == '.parquet':
                        df = pd.read_parquet(cleaned_data_file)
                    else:
                        df = pd.read_csv(cleaned_data_file)
                    
                    data_info = {
                        "source_file": str(cleaned_data_file),
                        "shape": df.shape,
                        "columns": list(df.columns),
                        "date_range": {
                            "start": str(df['date'].min()) if 'date' in df.columns else "Unknown",
                            "end": str(df['date'].max()) if 'date' in df.columns else "Unknown"
                        },
                        "tickers": list(df['ticker'].unique()) if 'ticker' in df.columns else [],
                        "num_tickers": df['ticker'].nunique() if 'ticker' in df.columns else 0,
                        "missing_values": df.isnull().sum().sum(),
                        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
                        "sample_data": df.head(3).to_dict() if len(df) > 0 else {}
                    }
                    
                    # Feature analysis
                    numeric_columns = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_columns) > 0:
                        data_info["feature_statistics"] = {
                            "numeric_features": len(numeric_columns),
                            "feature_ranges": {
                                col: {"min": float(df[col].min()), "max": float(df[col].max()), "mean": float(df[col].mean())}
                                for col in numeric_columns[:10]  # First 10 features only
                            }
                        }
            
            except Exception as e:
                data_info = {"error": f"Could not analyze data: {str(e)}"}
        else:
            data_info = {"error": "No data files found"}
        
        # Save data info
        with open(self.archive_dir / "dataset_info.json", 'w') as f:
            json.dump(convert_numpy_types(data_info, f, indent=2))
        
        return data_info
    
    def parse_training_logs(self):
        """Parse and analyze training logs."""
        print("📋 Parsing training logs...")
        
        log_files = list(Path("logs").glob("*.log")) if Path("logs").exists() else []
        if not log_files:
            return {"error": "No log files found"}
        
        latest_log = max(log_files, key=os.path.getctime)
        
        metrics = {
            "log_file": str(latest_log),
            "training_metrics": {
                "episode_rewards": [],
                "vae_losses": [],
                "iterations": [],
                "timestamps": []
            },
            "training_events": [],
            "errors": [],
            "warnings": []
        }
        
        try:
            with open(latest_log, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    # Extract rewards
                    if "Episode reward:" in line:
                        try:
                            reward = float(line.split("Episode reward:")[1].split()[0])
                            metrics["training_metrics"]["episode_rewards"].append(reward)
                        except:
                            pass
                    
                    # Extract VAE losses
                    if "VAE loss:" in line:
                        try:
                            loss = float(line.split("VAE loss:")[1].split()[0])
                            metrics["training_metrics"]["vae_losses"].append(loss)
                        except:
                            pass
                    
                    # Extract iterations
                    if "Iteration" in line and ":" in line:
                        try:
                            iter_num = int(line.split("Iteration")[1].split(":")[0].strip())
                            metrics["training_metrics"]["iterations"].append(iter_num)
                        except:
                            pass
                    
                    # Extract important events
                    if any(keyword in line.lower() for keyword in ["started", "completed", "checkpoint", "evaluation"]):
                        metrics["training_events"].append({
                            "line": line_num,
                            "event": line.strip()
                        })
                    
                    # Extract errors and warnings
                    if "ERROR" in line or "Error" in line:
                        metrics["errors"].append({
                            "line": line_num,
                            "error": line.strip()
                        })
                    
                    if "WARNING" in line or "Warning" in line:
                        metrics["warnings"].append({
                            "line": line_num,
                            "warning": line.strip()
                        })
            
            # Calculate summary statistics
            rewards = metrics["training_metrics"]["episode_rewards"]
            vae_losses = metrics["training_metrics"]["vae_losses"]
            
            if rewards:
                metrics["summary"] = {
                    "total_episodes": len(rewards),
                    "avg_reward": float(np.mean(rewards)),
                    "best_reward": float(np.max(rewards)),
                    "worst_reward": float(np.min(rewards)),
                    "final_reward": float(rewards[-1]),
                    "reward_std": float(np.std(rewards)),
                    "reward_trend": "improving" if len(rewards) > 10 and np.mean(rewards[-10:]) > np.mean(rewards[:10]) else "declining"
                }
            
            if vae_losses:
                metrics["summary"]["avg_vae_loss"] = float(np.mean(vae_losses))
                metrics["summary"]["final_vae_loss"] = float(vae_losses[-1])
                metrics["summary"]["vae_learning"] = "converged" if len(vae_losses) > 10 and vae_losses[-1] < np.mean(vae_losses[:10]) else "not_converged"
            
        except Exception as e:
            metrics["parsing_error"] = str(e)
        
        # Save metrics
        with open(self.archive_dir / "training_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def create_comprehensive_analysis(self, metrics):
        """Create comprehensive analysis plots and report."""
        print("📈 Creating comprehensive analysis...")
        
        # Create analysis plots
        if metrics.get("training_metrics"):
            rewards = metrics["training_metrics"]["episode_rewards"]
            vae_losses = metrics["training_metrics"]["vae_losses"]
            
            if rewards or vae_losses:
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                fig.suptitle(f'VariBAD Training Analysis - {self.run_name}', fontsize=16, fontweight='bold')
                
                # 1. Reward progression
                if rewards:
                    axes[0,0].plot(rewards, 'b-', alpha=0.6, linewidth=1, label='Episode Rewards')
                    if len(rewards) > 10:
                        smoothed = pd.Series(rewards).rolling(window=min(20, len(rewards)//5)).mean()
                        axes[0,0].plot(smoothed, 'r-', linewidth=3, label='Smoothed (Moving Avg)')
                    axes[0,0].set_title('Portfolio Performance (Episode Rewards)', fontweight='bold')
                    axes[0,0].set_xlabel('Episode')
                    axes[0,0].set_ylabel('DSR Reward')
                    axes[0,0].grid(True, alpha=0.3)
                    axes[0,0].legend()
                    
                    # Add performance annotations
                    if len(rewards) > 0:
                        final_reward = rewards[-1]
                        avg_reward = np.mean(rewards)
                        axes[0,0].axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Break-even')
                        axes[0,0].text(0.02, 0.98, f'Final: {final_reward:.4f}\nAverage: {avg_reward:.4f}', 
                                     transform=axes[0,0].transAxes, verticalalignment='top',
                                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
                
                # 2. VAE Loss
                if vae_losses:
                    axes[0,1].plot(vae_losses, 'g-', alpha=0.7, linewidth=2)
                    if len(vae_losses) > 10:
                        smoothed = pd.Series(vae_losses).rolling(window=min(10, len(vae_losses)//3)).mean()
                        axes[0,1].plot(smoothed, 'r-', linewidth=3, label='Smoothed')
                    axes[0,1].set_title('VAE Learning (Regime Detection)', fontweight='bold')
                    axes[0,1].set_xlabel('Update')
                    axes[0,1].set_ylabel('Loss')
                    axes[0,1].grid(True, alpha=0.3)
                    if len(vae_losses) > 10:
                        axes[0,1].legend()
                
                # 3. Reward distribution
                if rewards:
                    axes[0,2].hist(rewards, bins=min(30, len(rewards)//5), alpha=0.7, color='skyblue', edgecolor='black')
                    axes[0,2].axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2,
                                     label=f'Mean: {np.mean(rewards):.4f}')
                    axes[0,2].axvline(0, color='gray', linestyle='-', alpha=0.5, label='Break-even')
                    axes[0,2].set_title('Reward Distribution', fontweight='bold')
                    axes[0,2].set_xlabel('DSR Reward')
                    axes[0,2].set_ylabel('Frequency')
                    axes[0,2].legend()
                    axes[0,2].grid(True, alpha=0.3)
                
                # 4. Learning curves comparison
                if rewards and len(rewards) > 20:
                    window_size = len(rewards) // 10
                    recent_performance = pd.Series(rewards).rolling(window=window_size).mean()
                    axes[1,0].plot(recent_performance, 'purple', linewidth=3)
                    axes[1,0].set_title('Learning Curve (Rolling Average)', fontweight='bold')
                    axes[1,0].set_xlabel('Episode')
                    axes[1,0].set_ylabel('Average Reward')
                    axes[1,0].grid(True, alpha=0.3)
                
                # 5. Performance phases
                if rewards and len(rewards) > 50:
                    # Divide training into phases
                    phase_size = len(rewards) // 4
                    phases = ['Early', 'Mid-Early', 'Mid-Late', 'Late']
                    phase_rewards = []
                    
                    for i in range(4):
                        start_idx = i * phase_size
                        end_idx = (i + 1) * phase_size if i < 3 else len(rewards)
                        phase_avg = np.mean(rewards[start_idx:end_idx])
                        phase_rewards.append(phase_avg)
                    
                    axes[1,1].bar(phases, phase_rewards, color=['lightcoral', 'lightblue', 'lightgreen', 'gold'])
                    axes[1,1].set_title('Performance by Training Phase', fontweight='bold')
                    axes[1,1].set_ylabel('Average Reward')
                    axes[1,1].grid(True, alpha=0.3)
                    axes[1,1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                
                # 6. Recent performance detail
                if rewards and len(rewards) > 20:
                    recent_count = min(50, len(rewards) // 3)
                    recent_rewards = rewards[-recent_count:]
                    axes[1,2].plot(range(len(rewards)-recent_count, len(rewards)), recent_rewards, 
                                  'b-', linewidth=2, marker='o', markersize=3)
                    axes[1,2].set_title(f'Recent Performance (Last {recent_count} Episodes)', fontweight='bold')
                    axes[1,2].set_xlabel('Episode')
                    axes[1,2].set_ylabel('Reward')
                    axes[1,2].grid(True, alpha=0.3)
                    axes[1,2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                
                # Save the comprehensive plot
                plot_path = self.archive_dir / "comprehensive_analysis.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                print(f"   📊 Saved comprehensive analysis: {plot_path}")
        
        # Create executive summary report
        self.create_executive_summary(metrics)
    
    def create_executive_summary(self, metrics):
        """Create an executive summary report."""
        print("📝 Creating executive summary...")
        
        summary_lines = []
        summary_lines.append(f"# VariBAD Training Results - Executive Summary")
        summary_lines.append(f"**Run Name:** {self.run_name}")
        summary_lines.append(f"**Date:** {self.timestamp}")
        summary_lines.append("")
        
        # Training Overview
        summary_lines.append("## Training Overview")
        summary_lines.append("")
        
        if "summary" in metrics:
            s = metrics["summary"]
            summary_lines.append(f"- **Total Episodes:** {s.get('total_episodes', 'Unknown')}")
            summary_lines.append(f"- **Average Reward:** {s.get('avg_reward', 'Unknown'):.4f}")
            summary_lines.append(f"- **Best Reward:** {s.get('best_reward', 'Unknown'):.4f}")
            summary_lines.append(f"- **Final Reward:** {s.get('final_reward', 'Unknown'):.4f}")
            summary_lines.append(f"- **Reward Trend:** {s.get('reward_trend', 'Unknown').title()}")
            summary_lines.append(f"- **VAE Learning:** {s.get('vae_learning', 'Unknown').replace('_', ' ').title()}")
        
        summary_lines.append("")
        
        # Performance Assessment
        summary_lines.append("## Performance Assessment")
        summary_lines.append("")
        
        if "summary" in metrics:
            avg_reward = metrics["summary"].get("avg_reward", 0)
            final_reward = metrics["summary"].get("final_reward", 0)
            trend = metrics["summary"].get("reward_trend", "unknown")
            
            if avg_reward > 0:
                performance = "🟢 **PROFITABLE** - Model generates positive returns"
            elif avg_reward > -0.05:
                performance = "🟡 **MARGINAL** - Model near break-even"
            else:
                performance = "🔴 **UNPROFITABLE** - Model loses money consistently"
            
            summary_lines.append(f"**Overall Performance:** {performance}")
            summary_lines.append("")
            
            if trend == "improving":
                trend_assessment = "🟢 **LEARNING** - Performance improves over time"
            elif trend == "declining":
                trend_assessment = "🔴 **DEGRADING** - Performance worsens over time"
            else:
                trend_assessment = "🟡 **STABLE** - No clear learning trend"
            
            summary_lines.append(f"**Learning Trend:** {trend_assessment}")
            summary_lines.append("")
        
        # Issues and Recommendations
        summary_lines.append("## Issues and Recommendations")
        summary_lines.append("")
        
        if metrics.get("errors"):
            summary_lines.append("### 🚨 Errors Detected:")
            for error in metrics["errors"][:5]:  # Show first 5 errors
                summary_lines.append(f"- Line {error['line']}: {error['error'][:100]}...")
            summary_lines.append("")
        
        if metrics.get("warnings"):
            summary_lines.append("### ⚠️ Warnings:")
            for warning in metrics["warnings"][:3]:  # Show first 3 warnings
                summary_lines.append(f"- Line {warning['line']}: {warning['warning'][:100]}...")
            summary_lines.append("")
        
        # Recommendations based on performance
        summary_lines.append("### 💡 Recommendations:")
        if "summary" in metrics:
            avg_reward = metrics["summary"].get("avg_reward", 0)
            vae_learning = metrics["summary"].get("vae_learning", "unknown")
            
            if avg_reward < 0:
                summary_lines.append("- **Policy Issue**: Negative returns suggest policy is not learning effectively")
                summary_lines.append("- **Try**: Longer episodes, different reward function, or policy learning rate adjustment")
            
            if vae_learning == "not_converged":
                summary_lines.append("- **VAE Issue**: VAE not converging suggests regime detection problems")
                summary_lines.append("- **Try**: More VAE updates, different latent dimension, or simpler data")
            
            if avg_reward < 0 and vae_learning == "converged":
                summary_lines.append("- **Integration Issue**: VAE learns but policy doesn't use regime info effectively")
                summary_lines.append("- **Try**: Check policy network architecture or belief integration mechanism")
        
        summary_lines.append("")
        
        # File Locations
        summary_lines.append("## Archive Contents")
        summary_lines.append("")
        summary_lines.append("This archive contains:")
        summary_lines.append("- `system_info.json` - Complete system and environment information")
        summary_lines.append("- `training_config.json` - Training configuration and parameters")
        summary_lines.append("- `dataset_info.json` - Dataset characteristics and statistics")
        summary_lines.append("- `training_metrics.json` - Complete training metrics and logs")
        summary_lines.append("- `comprehensive_analysis.png` - Visual analysis of training progress")
        summary_lines.append("- `executive_summary.md` - This summary document")
        summary_lines.append("- `model_checkpoints/` - Saved model states (if available)")
        summary_lines.append("- `raw_logs/` - Original log files")
        summary_lines.append("")
        
        summary_lines.append("---")
        summary_lines.append(f"*Generated automatically by VariBAD Results Archiver on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        # Save executive summary
        summary_path = self.archive_dir / "executive_summary.md"
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"   📝 Saved executive summary: {summary_path}")
    
    def copy_artifacts(self):
        """Copy all relevant training artifacts to archive."""
        print("📁 Copying training artifacts...")
        
        # Copy checkpoints
        if Path("checkpoints").exists():
            checkpoint_dir = self.archive_dir / "model_checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            
            for checkpoint in Path("checkpoints").glob("*.pt"):
                shutil.copy2(checkpoint, checkpoint_dir / checkpoint.name)
                print(f"   Copied checkpoint: {checkpoint.name}")
        
        # Copy logs
        if Path("logs").exists():
            logs_dir = self.archive_dir / "raw_logs"
            logs_dir.mkdir(exist_ok=True)
            
            for log_file in Path("logs").glob("*.log"):
                shutil.copy2(log_file, logs_dir / log_file.name)
                print(f"   Copied log: {log_file.name}")
        
        # Copy plots (if any)
        if Path("plots").exists():
            plots_dir = self.archive_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            for plot in Path("plots").glob("*.png"):
                shutil.copy2(plot, plots_dir / plot.name)
                print(f"   Copied plot: {plot.name}")
        
        # Copy any results files
        if Path("results").exists():
            results_dir = self.archive_dir / "results"
            results_dir.mkdir(exist_ok=True)
            
            for result in Path("results").glob("*"):
                if result.is_file():
                    shutil.copy2(result, results_dir / result.name)
                    print(f"   Copied result: {result.name}")
    
    def create_zip_archive(self):
        """Create a zip file of the complete archive."""
        print("🗜️  Creating zip archive...")
        
        zip_path = f"archives/{self.run_name}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in self.archive_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.archive_dir.parent)
                    zipf.write(file_path, arcname)
        
        zip_size = os.path.getsize(zip_path) / 1024 / 1024  # MB
        print(f"   📦 Created zip archive: {zip_path} ({zip_size:.1f} MB)")
        
        return zip_path
    
    def create_complete_archive(self):
        """Create complete training results archive."""
        print(f"\n🚀 Creating Complete VariBAD Training Archive")
        print("=" * 60)
        
        # Capture all information
        system_info = self.capture_system_info()
        config = self.capture_training_config()
        data_info = self.capture_data_info()
        metrics = self.parse_training_logs()
        
        # Create analysis
        self.create_comprehensive_analysis(metrics)
        
        # Copy artifacts
        self.copy_artifacts()
        
        # Create zip
        zip_path = self.create_zip_archive()
        
        print("\n" + "=" * 60)
        print("✅ ARCHIVE COMPLETE!")
        print("=" * 60)
        print(f"📁 Archive directory: {self.archive_dir}")
        print(f"📦 Zip archive: {zip_path}")
        print(f"📝 Executive summary: {self.archive_dir}/executive_summary.md")
        print(f"📊 Analysis plots: {self.archive_dir}/comprehensive_analysis.png")
        print("\n🎯 This archive contains everything needed to understand this training run!")
        
        return {
            "archive_dir": str(self.archive_dir),
            "zip_path": zip_path,
            "run_name": self.run_name,
            "timestamp": self.timestamp
        }

def main():
    """Main function to create training archive."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Archive VariBAD training results")
    parser.add_argument('--name', type=str, help='Custom name for this training run')
    parser.add_argument('--description', type=str, help='Description of this training run')
    
    args = parser.parse_args()
    
    # Create custom run name if provided
    run_name = args.name
    if args.description:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{args.name or 'varibad'}_{timestamp}"
    
    # Create archiver and run
    archiver = VariBADResultsArchiver(run_name=run_name)
    
    # Add description to archive if provided
    if args.description:
        desc_file = archiver.archive_dir / "run_description.txt"
        with open(desc_file, 'w') as f:
            f.write(f"Training Run Description\n")
            f.write(f"========================\n\n")
            f.write(f"Name: {args.name or 'Unnamed'}\n")
            f.write(f"Date: {archiver.timestamp}\n")
            f.write(f"Description: {args.description}\n")
    
    # Create complete archive
    result = archiver.create_complete_archive()
    
    return result

if __name__ == "__main__":
    main()