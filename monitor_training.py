#!/usr/bin/env python3
"""
Enhanced VariBAD Training Monitor and Visualizer
Real-time monitoring with comprehensive plots and analysis
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import os
import re
import time
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import seaborn as sns
import json
from collections import defaultdict

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

class EnhancedVariBADMonitor:
    def __init__(self, log_dir="logs", checkpoint_dir="checkpoints", plots_dir="plots", update_interval=30):
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.plots_dir = Path(plots_dir)
        self.update_interval = update_interval
        
        # Create directories
        self.plots_dir.mkdir(exist_ok=True)
        
        # Statistics tracking
        self.training_start_time = None
        self.last_update = None
        
        print(f"🔍 Enhanced VariBAD Training Monitor")
        print(f"   Log directory: {self.log_dir}")
        print(f"   Checkpoint directory: {self.checkpoint_dir}")
        print(f"   Plots directory: {self.plots_dir}")
        print(f"   Update interval: {update_interval}s")
    
    def parse_logs(self):
        """Enhanced log parsing with more metrics."""
        log_files = list(self.log_dir.glob("varibad_pipeline_*.log"))
        
        if not log_files:
            return None, None
        
        # Get the most recent log file
        latest_log = max(log_files, key=os.path.getctime)
        
        # Parse training start time from filename
        try:
            timestamp_str = latest_log.stem.split('_')[-2] + '_' + latest_log.stem.split('_')[-1]
            self.training_start_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        except:
            self.training_start_time = datetime.fromtimestamp(os.path.getctime(latest_log))
        
        print(f"📄 Parsing log: {latest_log}")
        
        metrics = defaultdict(list)
        
        with open(latest_log, 'r') as f:
            lines = f.readlines()
        
        current_iter = None
        
        for line in lines:
            # Parse iteration markers
            if "Iteration " in line and ":" in line:
                match = re.search(r"Iteration (\d+):", line)
                if match:
                    current_iter = int(match.group(1))
            
            # Parse various metrics with timestamps
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            timestamp = None
            if timestamp_match:
                try:
                    timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                except:
                    pass
            
            # Episode rewards
            if "Episode reward:" in line and current_iter is not None:
                match = re.search(r"Episode reward: ([-+]?\d*\.?\d+)", line)
                if match:
                    metrics['iteration'].append(current_iter)
                    metrics['avg_episode_reward'].append(float(match.group(1)))
                    metrics['timestamp'].append(timestamp)
            
            # VAE metrics
            if "VAE loss:" in line:
                match = re.search(r"VAE loss: ([-+]?\d*\.?\d+)", line)
                if match:
                    metrics['avg_vae_loss'].append(float(match.group(1)))
            
            if "ELBO:" in line:
                match = re.search(r"ELBO: ([-+]?\d*\.?\d+)", line)
                if match:
                    metrics['avg_elbo'].append(float(match.group(1)))
            
            # Policy metrics
            if "policy_loss:" in line:
                match = re.search(r"policy_loss.*?: ([-+]?\d*\.?\d+)", line)
                if match:
                    metrics['policy_loss'].append(float(match.group(1)))
            
            # Buffer statistics
            if "Buffer episodes:" in line:
                match = re.search(r"Buffer episodes: (\d+)", line)
                if match:
                    metrics['buffer_episodes'].append(int(match.group(1)))
            
            # Evaluation metrics
            eval_metrics = ['eval_total_reward', 'eval_average_reward', 'eval_total_return', 'eval_sharpe_ratio']
            for metric in eval_metrics:
                if metric in line:
                    match = re.search(rf"{metric}.*?: ([-+]?\d*\.?\d+)", line)
                    if match:
                        metrics[metric].append(float(match.group(1)))
            
            # Training progress indicators
            if "Training completed!" in line:
                metrics['training_completed'] = True
            
            if "Starting VariBAD training" in line:
                metrics['training_started'] = True
        
        # Convert to DataFrame
        if not metrics or len(metrics.get('iteration', [])) == 0:
            return None, latest_log
        
        # Align all metrics to iterations
        max_iter_count = len(metrics['iteration'])
        
        df_dict = {}
        for key, values in metrics.items():
            if key in ['training_completed', 'training_started']:
                continue
            
            if isinstance(values, list) and len(values) > 0:
                if len(values) == max_iter_count:
                    df_dict[key] = values
                elif len(values) < max_iter_count:
                    # Pad with last known value or NaN
                    padded = values + [values[-1] if values else np.nan] * (max_iter_count - len(values))
                    df_dict[key] = padded[:max_iter_count]
                else:
                    # Truncate to match
                    df_dict[key] = values[:max_iter_count]
            else:
                df_dict[key] = [np.nan] * max_iter_count
        
        df = pd.DataFrame(df_dict)
        
        # Calculate derived metrics
        if len(df) > 1:
            df['iteration_time'] = df['timestamp'].diff().dt.total_seconds() / 60  # minutes per iteration
            df['cumulative_time'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 3600  # hours
            df['iterations_per_hour'] = 1 / (df['iteration_time'] / 60)
            
            # Smoothed metrics
            window = min(10, len(df))
            df['reward_smooth'] = df['avg_episode_reward'].rolling(window=window, min_periods=1).mean()
            df['vae_loss_smooth'] = df['avg_vae_loss'].rolling(window=window, min_periods=1).mean()
        
        return df, latest_log
    
    def create_comprehensive_plot(self, metrics_df, save=True):
        """Create comprehensive training dashboard."""
        if metrics_df is None or len(metrics_df) == 0:
            print("⚠️  No metrics to plot")
            return None
        
        # Create large dashboard
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Main title with status
        training_status = "🏃 Training in Progress"
        if metrics_df.get('training_completed', False):
            training_status = "✅ Training Completed"
        
        elapsed_time = "Unknown"
        if self.training_start_time:
            elapsed = datetime.now() - self.training_start_time
            hours, remainder = divmod(elapsed.total_seconds(), 3600)
            minutes, _ = divmod(remainder, 60)
            elapsed_time = f"{int(hours):02d}:{int(minutes):02d}"
        
        fig.suptitle(f'VariBAD Training Dashboard - {training_status} - Elapsed: {elapsed_time}', 
                     fontsize=18, fontweight='bold')
        
        # 1. Episode Rewards (main plot)
        ax1 = fig.add_subplot(gs[0, :2])
        if 'avg_episode_reward' in metrics_df.columns and not metrics_df['avg_episode_reward'].isna().all():
            ax1.plot(metrics_df['iteration'], metrics_df['avg_episode_reward'], 
                    'b-', linewidth=1, alpha=0.7, label='Episode Reward')
            if 'reward_smooth' in metrics_df.columns:
                ax1.plot(metrics_df['iteration'], metrics_df['reward_smooth'],
                        'b-', linewidth=3, label='Smoothed Reward')
            ax1.set_title('Portfolio Performance (Episode Rewards)', fontweight='bold')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('DSR Reward')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # 2. VAE Learning Progress
        ax2 = fig.add_subplot(gs[0, 2:])
        if 'avg_vae_loss' in metrics_df.columns and not metrics_df['avg_vae_loss'].isna().all():
            ax2.plot(metrics_df['iteration'], metrics_df['avg_vae_loss'], 
                    'r-', linewidth=1, alpha=0.7, label='VAE Loss')
            if 'vae_loss_smooth' in metrics_df.columns:
                ax2.plot(metrics_df['iteration'], metrics_df['vae_loss_smooth'],
                        'r-', linewidth=3, label='Smoothed Loss')
            ax2.set_title('VAE Learning (Regime Detection)', fontweight='bold')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Loss')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # 3. Training Speed
        ax3 = fig.add_subplot(gs[1, 0])
        if 'iterations_per_hour' in metrics_df.columns and not metrics_df['iterations_per_hour'].isna().all():
            ax3.plot(metrics_df['iteration'], metrics_df['iterations_per_hour'], 'g-', linewidth=2)
            avg_speed = metrics_df['iterations_per_hour'].mean()
            ax3.axhline(y=avg_speed, color='g', linestyle='--', alpha=0.5, label=f'Avg: {avg_speed:.1f}/hr')
            ax3.set_title('Training Speed')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Iterations/Hour')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # 4. Buffer Growth
        ax4 = fig.add_subplot(gs[1, 1])
        if 'buffer_episodes' in metrics_df.columns and not metrics_df['buffer_episodes'].isna().all():
            ax4.plot(metrics_df['iteration'], metrics_df['buffer_episodes'], 'purple', linewidth=2)
            ax4.set_title('Experience Buffer')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Episodes Stored')
            ax4.grid(True, alpha=0.3)
        
        # 5. Policy Performance
        ax5 = fig.add_subplot(gs[1, 2])
        if 'policy_loss' in metrics_df.columns and not metrics_df['policy_loss'].isna().all():
            ax5.plot(metrics_df['iteration'][1:len(metrics_df['policy_loss'])+1], 
                    metrics_df['policy_loss'], 'orange', linewidth=2)
            ax5.set_title('Policy Learning')
            ax5.set_xlabel('Iteration')
            ax5.set_ylabel('Policy Loss')
            ax5.grid(True, alpha=0.3)
        
        # 6. Evaluation Results
        ax6 = fig.add_subplot(gs[1, 3])
        eval_cols = [col for col in metrics_df.columns if col.startswith('eval_') and not metrics_df[col].isna().all()]
        if eval_cols:
            for i, col in enumerate(eval_cols[:3]):  # Show top 3 eval metrics
                data = metrics_df[col].dropna()
                if len(data) > 0:
                    ax6.plot(range(len(data)), data, 'o-', label=col.replace('eval_', ''), alpha=0.8)
            ax6.set_title('Evaluation Metrics')
            ax6.set_xlabel('Evaluation #')
            ax6.set_ylabel('Performance')
            ax6.legend(fontsize=8)
            ax6.grid(True, alpha=0.3)
        
        # 7. Training Statistics (Text Summary)
        ax7 = fig.add_subplot(gs[2, :2])
        stats_text = self._generate_stats_summary(metrics_df)
        ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        ax7.set_title('Training Statistics', fontweight='bold')
        ax7.axis('off')
        
        # 8. Performance Distribution
        ax8 = fig.add_subplot(gs[2, 2])
        if 'avg_episode_reward' in metrics_df.columns and not metrics_df['avg_episode_reward'].isna().all():
            rewards = metrics_df['avg_episode_reward'].dropna()
            ax8.hist(rewards, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax8.axvline(rewards.mean(), color='red', linestyle='--', label=f'Mean: {rewards.mean():.4f}')
            ax8.set_title('Reward Distribution')
            ax8.set_xlabel('Episode Reward')
            ax8.set_ylabel('Frequency')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        # 9. Recent Progress (last 50 iterations)
        ax9 = fig.add_subplot(gs[2, 3])
        recent_data = metrics_df.tail(50)
        if len(recent_data) > 1 and 'avg_episode_reward' in recent_data.columns:
            ax9.plot(recent_data['iteration'], recent_data['avg_episode_reward'], 'b-', linewidth=2)
            ax9.set_title('Recent Progress (Last 50)')
            ax9.set_xlabel('Iteration')
            ax9.set_ylabel('Reward')
            ax9.grid(True, alpha=0.3)
        
        # 10. System Health (bottom row)
        ax10 = fig.add_subplot(gs[3, :])
        health_text = self._generate_health_report()
        ax10.text(0.05, 0.95, health_text, transform=ax10.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        ax10.set_title('System Health & Recommendations', fontweight='bold')
        ax10.axis('off')
        
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = self.plots_dir / f'varibad_dashboard_{timestamp}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"📊 Dashboard saved: {filename}")
        
        return fig
    
    def _generate_stats_summary(self, df):
        """Generate comprehensive statistics summary."""
        stats = []
        stats.append("📊 TRAINING STATISTICS")
        stats.append("=" * 25)
        
        # Basic info
        stats.append(f"Total Iterations: {len(df)}")
        
        if 'avg_episode_reward' in df.columns:
            rewards = df['avg_episode_reward'].dropna()
            if len(rewards) > 0:
                stats.append(f"Latest Reward: {rewards.iloc[-1]:.4f}")
                stats.append(f"Best Reward: {rewards.max():.4f}")
                stats.append(f"Average Reward: {rewards.mean():.4f}")
                stats.append(f"Reward Std: {rewards.std():.4f}")
        
        # VAE learning
        if 'avg_vae_loss' in df.columns:
            vae_loss = df['avg_vae_loss'].dropna()
            if len(vae_loss) > 0:
                stats.append(f"Latest VAE Loss: {vae_loss.iloc[-1]:.4f}")
                stats.append(f"Best VAE Loss: {vae_loss.min():.4f}")
        
        # Training speed
        if 'iterations_per_hour' in df.columns:
            speed = df['iterations_per_hour'].dropna()
            if len(speed) > 0:
                stats.append(f"Training Speed: {speed.mean():.1f} iter/hr")
        
        # Buffer status
        if 'buffer_episodes' in df.columns:
            buffer = df['buffer_episodes'].dropna()
            if len(buffer) > 0:
                stats.append(f"Buffer Size: {buffer.iloc[-1]} episodes")
        
        # Time estimates
        if len(df) > 0 and 'iterations_per_hour' in df.columns:
            speed = df['iterations_per_hour'].dropna()
            if len(speed) > 0:
                avg_speed = speed.mean()
                stats.append("")
                stats.append("🕐 TIME ESTIMATES")
                stats.append("-" * 15)
                remaining_1000 = max(0, 1000 - len(df))
                remaining_2000 = max(0, 2000 - len(df))
                if remaining_1000 > 0:
                    hours_1000 = remaining_1000 / avg_speed
                    stats.append(f"To 1000 iter: {hours_1000:.1f}h")
                if remaining_2000 > 0:
                    hours_2000 = remaining_2000 / avg_speed
                    stats.append(f"To 2000 iter: {hours_2000:.1f}h")
        
        return '\n'.join(stats)
    
    def _generate_health_report(self):
        """Generate system health and recommendations."""
        health = []
        health.append("🏥 SYSTEM HEALTH & RECOMMENDATIONS")
        health.append("=" * 40)
        
        # Check GPU status
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                memory_allocated = torch.cuda.memory_allocated(0) / 1e9
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                health.append(f"🔥 GPU: {gpu_name}")
                health.append(f"   Memory: {memory_allocated:.1f}/{memory_total:.1f} GB ({memory_allocated/memory_total*100:.1f}%)")
            else:
                health.append("💻 Device: CPU (consider GPU for faster training)")
        except:
            health.append("❓ Device status unknown")
        
        # Check log file size
        try:
            log_files = list(self.log_dir.glob("*.log"))
            if log_files:
                latest_log = max(log_files, key=os.path.getctime)
                log_size_mb = os.path.getsize(latest_log) / 1e6
                health.append(f"📄 Log size: {log_size_mb:.1f} MB")
        except:
            pass
        
        # Check checkpoint directory
        try:
            checkpoints = list(self.checkpoint_dir.glob("*.pt"))
            if checkpoints:
                total_size = sum(os.path.getsize(f) for f in checkpoints) / 1e6
                health.append(f"💾 Checkpoints: {len(checkpoints)} files ({total_size:.1f} MB)")
        except:
            pass
        
        # Recommendations
        health.append("")
        health.append("💡 RECOMMENDATIONS")
        health.append("-" * 18)
        health.append("• Monitor GPU memory if training slows")
        health.append("• Save plots regularly for progress tracking")
        health.append("• Use tmux for long training sessions")
        health.append("• Check logs if training stalls")
        
        # Last update time
        health.append("")
        health.append(f"🕒 Last updated: {datetime.now().strftime('%H:%M:%S')}")
        
        return '\n'.join(health)
    
    def monitor_realtime(self):
        """Enhanced real-time monitoring loop."""
        print(f"🔄 Starting enhanced real-time monitoring")
        print(f"📊 Updates every {self.update_interval}s")
        print(f"💾 Plots saved to: {self.plots_dir}")
        print("Press Ctrl+C to stop")
        
        try:
            iteration_count = 0
            while True:
                iteration_count += 1
                
                print(f"\n{'='*80}")
                print(f"📊 Enhanced Monitor Update #{iteration_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*80}")
                
                # Parse logs and create dashboard
                metrics_df, latest_log = self.parse_logs()
                
                if metrics_df is not None and len(metrics_df) > 0:
                    # Create and save dashboard
                    fig = self.create_comprehensive_plot(metrics_df, save=True)
                    
                    if fig:
                        plt.show(block=False)
                        plt.pause(2)
                        plt.close(fig)
                    
                    # Print latest metrics to console
                    self._print_latest_metrics(metrics_df)
                    
                else:
                    print("⏳ Waiting for training data...")
                    if latest_log:
                        print(f"📄 Monitoring: {latest_log}")
                
                # Check for checkpoints
                self._print_checkpoint_status()
                
                # System status
                self._print_system_status()
                
                # Wait for next update
                print(f"\n⏰ Next update in {self.update_interval}s...")
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Enhanced monitoring stopped")
            print(f"📊 Generated {iteration_count} dashboard updates")
    
    def _print_latest_metrics(self, df):
        """Print latest metrics to console."""
        if len(df) == 0:
            return
        
        latest = df.iloc[-1]
        print(f"📈 Latest Metrics (Iteration {latest.get('iteration', 'N/A')}):")
        
        if 'avg_episode_reward' in df.columns and not pd.isna(latest['avg_episode_reward']):
            print(f"   💰 Reward: {latest['avg_episode_reward']:.4f}")
            
        if 'avg_vae_loss' in df.columns and not pd.isna(latest['avg_vae_loss']):
            print(f"   🧠 VAE Loss: {latest['avg_vae_loss']:.4f}")
            
        if 'buffer_episodes' in df.columns and not pd.isna(latest['buffer_episodes']):
            print(f"   📚 Buffer: {int(latest['buffer_episodes'])} episodes")
            
        if 'iterations_per_hour' in df.columns and not pd.isna(latest['iterations_per_hour']):
            print(f"   ⚡ Speed: {latest['iterations_per_hour']:.1f} iter/hour")
    
    def _print_checkpoint_status(self):
        """Print checkpoint information."""
        try:
            checkpoints = list(self.checkpoint_dir.glob("*.pt"))
            if checkpoints:
                latest_ckpt = max(checkpoints, key=os.path.getctime)
                ckpt_time = datetime.fromtimestamp(os.path.getctime(latest_ckpt))
                print(f"💾 Latest checkpoint: {latest_ckpt.name} ({ckpt_time.strftime('%H:%M:%S')})")
            else:
                print("💾 No checkpoints found yet")
        except Exception as e:
            print(f"💾 Checkpoint check failed: {e}")
    
    def _print_system_status(self):
        """Print system status."""
        try:
            import torch
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(0) / 1e9
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                utilization = memory_used / memory_total * 100
                print(f"🔥 GPU Memory: {memory_used:.1f}/{memory_total:.1f} GB ({utilization:.1f}%)")
            else:
                print("💻 Running on CPU")
        except Exception as e:
            print(f"🖥️  System status check failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced VariBAD Training Monitor")
    parser.add_argument('--mode', choices=['realtime', 'plot', 'checkpoints'], default='plot',
                       help='Monitoring mode')
    parser.add_argument('--interval', type=int, default=60,
                       help='Update interval for real-time monitoring (seconds)')
    parser.add_argument('--log-dir', default='logs', help='Log directory')
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--plots-dir', default='plots', help='Plots output directory')
    
    args = parser.parse_args()
    
    monitor = EnhancedVariBADMonitor(
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        plots_dir=args.plots_dir,
        update_interval=args.interval
    )
    
    if args.mode == 'realtime':
        monitor.monitor_realtime()
    elif args.mode == 'plot':
        print("📊 Generating training dashboard...")
        metrics_df, log_file = monitor.parse_logs()
        if metrics_df is not None:
            fig = monitor.create_comprehensive_plot(metrics_df, save=True)
            if fig:
                plt.show()
                print("✅ Dashboard displayed and saved")
            else:
                print("❌ Failed to create dashboard")
        else:
            print("⚠️  No training data found to plot")
            if log_file:
                print(f"📄 Checked log file: {log_file}")
    elif args.mode == 'checkpoints':
        monitor._print_checkpoint_status()

if __name__ == "__main__":
    main()