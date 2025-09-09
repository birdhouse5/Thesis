import torch
import psutil
import time
import gc
import threading
import queue
from typing import Optional, Dict, Any, List
import logging
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class ResourceLimits:
    """Resource limits for experiment execution."""
    max_memory_mb: Optional[int] = None
    max_gpu_memory_mb: Optional[int] = None
    max_runtime_minutes: Optional[int] = None
    memory_warning_threshold: float = 0.8  # Warn at 80% usage
    cleanup_frequency: int = 5  # Cleanup every N experiments

class ResourceMonitor:
    """Real-time resource monitoring during experiments."""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_queue = queue.Queue()
        self.peak_memory = 0
        self.peak_gpu_memory = 0
        
    def start_monitoring(self):
        """Start resource monitoring thread."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.debug("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        logger.debug("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # CPU and memory
                process = psutil.Process()
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024
                system_memory_percent = psutil.virtual_memory().percent
                
                # GPU memory if available
                gpu_memory_mb = 0
                if torch.cuda.is_available():
                    gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                    self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_memory_mb)
                
                self.peak_memory = max(self.peak_memory, memory_mb)
                
                # Check limits and warnings
                self._check_resource_limits(memory_mb, gpu_memory_mb, system_memory_percent)
                
                # Store metrics
                metrics = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'gpu_memory_mb': gpu_memory_mb,
                    'system_memory_percent': system_memory_percent
                }
                
                if not self.metrics_queue.full():
                    self.metrics_queue.put(metrics)
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                time.sleep(5)  # Back off on errors
    
    def _check_resource_limits(self, memory_mb: float, gpu_memory_mb: float, system_memory_percent: float):
        """Check if resource limits are exceeded."""
        
        # Memory warnings
        if system_memory_percent > self.limits.memory_warning_threshold * 100:
            logger.warning(f"High system memory usage: {system_memory_percent:.1f}%")
        
        if self.limits.max_memory_mb and memory_mb > self.limits.max_memory_mb:
            logger.warning(f"Process memory limit exceeded: {memory_mb:.1f}MB > {self.limits.max_memory_mb}MB")
        
        if self.limits.max_gpu_memory_mb and gpu_memory_mb > self.limits.max_gpu_memory_mb:
            logger.warning(f"GPU memory limit exceeded: {gpu_memory_mb:.1f}MB > {self.limits.max_gpu_memory_mb}MB")
    
    def get_peak_usage(self) -> Dict[str, float]:
        """Get peak resource usage."""
        return {
            'peak_memory_mb': self.peak_memory,
            'peak_gpu_memory_mb': self.peak_gpu_memory
        }
    
    def get_recent_metrics(self, last_n_seconds: int = 60) -> List[Dict[str, Any]]:
        """Get recent monitoring metrics."""
        cutoff_time = time.time() - last_n_seconds
        recent_metrics = []
        
        # Drain queue and filter recent metrics
        while not self.metrics_queue.empty():
            try:
                metric = self.metrics_queue.get_nowait()
                if metric['timestamp'] >= cutoff_time:
                    recent_metrics.append(metric)
            except queue.Empty:
                break
        
        return recent_metrics

class ResourceOptimizer:
    """Optimize resource usage during experiments."""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.cleanup_count = 0
        
    def optimize_before_experiment(self, experiment_name: str):
        """Optimize resources before starting an experiment."""
        logger.debug(f"Pre-experiment optimization for: {experiment_name}")
        
        # Always do cleanup before each experiment
        self.cleanup_resources()
        
        # Set optimal torch settings
        self.optimize_torch_settings()
        
        # Log current resource state
        self.log_resource_state("pre_experiment")
    
    def optimize_after_experiment(self, experiment_name: str):
        """Optimize resources after completing an experiment."""
        logger.debug(f"Post-experiment optimization for: {experiment_name}")
        
        # Always cleanup after each experiment
        self.cleanup_resources()
        
        # Periodic deep cleanup
        self.cleanup_count += 1
        if self.cleanup_count % self.limits.cleanup_frequency == 0:
            self.deep_cleanup()
        
        # Log final resource state
        self.log_resource_state("post_experiment")
    
    def cleanup_resources(self):
        """Standard resource cleanup."""
        # Python garbage collection
        collected = gc.collect()
        logger.debug(f"Garbage collected {collected} objects")
        
        # CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("CUDA cache cleared")
        
        # Small delay to ensure cleanup completion
        time.sleep(0.5)
    
    def deep_cleanup(self):
        """More aggressive cleanup performed periodically."""
        logger.info("Performing deep resource cleanup...")
        
        # Force multiple GC cycles
        for _ in range(3):
            gc.collect()
        
        # Clear all CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # Force Python to release memory back to OS
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except:
            pass  # Not available on all systems
        
        logger.info("Deep cleanup completed")
        time.sleep(1)
    
    def optimize_torch_settings(self):
        """Set optimal PyTorch settings for resource efficiency."""
        
        # Optimize for single GPU usage
        if torch.cuda.is_available():
            # Enable memory efficient attention if available
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except:
                pass
            
            # Optimize CUDA settings for memory efficiency
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = False  # More deterministic, less memory
            torch.backends.cudnn.deterministic = True
        
        # Set number of threads for CPU operations
        cpu_count = psutil.cpu_count(logical=False)  # Physical cores only
        optimal_threads = min(4, cpu_count)  # Don't use too many threads
        torch.set_num_threads(optimal_threads)
        
        logger.debug(f"Torch settings optimized: {optimal_threads} threads")
    
    def log_resource_state(self, phase: str):
        """Log current resource usage."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            system_memory = psutil.virtual_memory()
            
            gpu_info = ""
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
                gpu_info = f", GPU: {gpu_memory_mb:.1f}MB allocated, {gpu_reserved_mb:.1f}MB reserved"
            
            logger.debug(f"Resources {phase}: Process {memory_mb:.1f}MB, "
                        f"System {system_memory.percent:.1f}% used{gpu_info}")
                        
        except Exception as e:
            logger.warning(f"Could not log resource state: {e}")

class ExperimentScheduler:
    """Intelligent scheduling of experiments based on resource requirements."""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.resource_history = {}
        
    def estimate_experiment_resources(self, experiment_config) -> Dict[str, float]:
        """Estimate resource requirements for an experiment."""
        
        # Base estimates (can be refined based on historical data)
        base_memory = 2000  # MB
        base_time = 30  # minutes
        
        # Adjust based on encoder type
        encoder_multipliers = {
            'vae': {'memory': 1.5, 'time': 1.3},
            'hmm': {'memory': 1.1, 'time': 0.9},
            'none': {'memory': 1.0, 'time': 1.0}
        }
        
        # Adjust based on asset class (crypto might be more intensive)
        asset_multipliers = {
            'sp500': {'memory': 1.0, 'time': 1.0},
            'crypto': {'memory': 1.2, 'time': 1.1}
        }
        
        encoder_mult = encoder_multipliers.get(experiment_config.encoder, encoder_multipliers['none'])
        asset_mult = asset_multipliers.get(experiment_config.asset_class, asset_multipliers['sp500'])
        
        estimated_memory = base_memory * encoder_mult['memory'] * asset_mult['memory']
        estimated_time = base_time * encoder_mult['time'] * asset_mult['time']
        
        return {
            'estimated_memory_mb': estimated_memory,
            'estimated_time_minutes': estimated_time,
            'confidence': 0.7  # Low confidence until we have historical data
        }
    
    def update_resource_history(self, experiment_name: str, actual_resources: Dict[str, float]):
        """Update historical resource usage data."""
        self.resource_history[experiment_name] = {
            **actual_resources,
            'timestamp': time.time()
        }
        
        # Keep only recent history (last 100 experiments)
        if len(self.resource_history) > 100:
            # Remove oldest entries
            sorted_history = sorted(self.resource_history.items(), key=lambda x: x[1]['timestamp'])
            self.resource_history = dict(sorted_history[-100:])
    
    def should_pause_before_experiment(self, experiment_config) -> bool:
        """Determine if we should pause before starting an experiment."""
        
        # Check current system load
        system_memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Pause if system is under high load
        if system_memory.percent > 85 or cpu_percent > 90:
            logger.warning(f"High system load detected (Memory: {system_memory.percent:.1f}%, "
                          f"CPU: {cpu_percent:.1f}%), pausing before next experiment")
            return True
        
        return False
    
    def get_pause_duration(self) -> int:
        """Get recommended pause duration in seconds."""
        return 30  # 30 second pause to let system recover

class ResourceManager:
    """Main resource management coordinator."""
    
    def __init__(self, 
                 limits: Optional[ResourceLimits] = None,
                 enable_monitoring: bool = True,
                 save_metrics: bool = True):
        
        self.limits = limits or ResourceLimits()
        self.monitor = ResourceMonitor(self.limits) if enable_monitoring else None
        self.optimizer = ResourceOptimizer(self.limits)
        self.scheduler = ExperimentScheduler(self.limits)
        self.save_metrics = save_metrics
        self.metrics_history = []
        
        if self.monitor and enable_monitoring:
            self.monitor.start_monitoring()
    
    def prepare_for_experiment(self, experiment_config, experiment_name: str):
        """Prepare system resources for an experiment."""
        
        # Check if we should pause
        if self.scheduler.should_pause_before_experiment(experiment_config):
            pause_duration = self.scheduler.get_pause_duration()
            logger.info(f"Pausing {pause_duration}s before experiment due to high system load")
            time.sleep(pause_duration)
        
        # Optimize resources
        self.optimizer.optimize_before_experiment(experiment_name)
        
        # Get resource estimates
        estimates = self.scheduler.estimate_experiment_resources(experiment_config)
        logger.debug(f"Estimated resources for {experiment_name}: "
                    f"{estimates['estimated_memory_mb']:.0f}MB, "
                    f"{estimates['estimated_time_minutes']:.1f}min")
        
        return estimates
    
    def finalize_experiment(self, experiment_name: str, actual_results: Dict[str, Any]):
        """Clean up after an experiment and update resource tracking."""
        
        # Extract actual resource usage
        actual_resources = {
            'memory_peak_mb': actual_results.get('memory_peak_mb', 0),
            'wall_time_minutes': actual_results.get('wall_time_seconds', 0) / 60,
            'gpu_memory_peak_mb': 0
        }
        
        # Get monitoring data if available
        if self.monitor:
            peak_usage = self.monitor.get_peak_usage()
            actual_resources.update(peak_usage)
        
        # Update scheduler with actual usage
        self.scheduler.update_resource_history(experiment_name, actual_resources)
        
        # Optimize resources after experiment
        self.optimizer.optimize_after_experiment(experiment_name)
        
        # Save metrics if enabled
        if self.save_metrics:
            self._save_resource_metrics(experiment_name, actual_resources)
    
    def _save_resource_metrics(self, experiment_name: str, resources: Dict[str, Any]):
        """Save resource metrics for analysis."""
        metrics_entry = {
            'experiment_name': experiment_name,
            'timestamp': time.time(),
            **resources
        }
        
        self.metrics_history.append(metrics_entry)
        
        # Periodically save to disk
        if len(self.metrics_history) % 10 == 0:
            metrics_file = Path("resource_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get summary of resource usage across all experiments."""
        if not self.metrics_history:
            return {}
        
        memory_values = [m.get('memory_peak_mb', 0) for m in self.metrics_history]
        time_values = [m.get('wall_time_minutes', 0) for m in self.metrics_history]
        gpu_values = [m.get('gpu_memory_peak_mb', 0) for m in self.metrics_history]
        
        return {
            'total_experiments': len(self.metrics_history),
            'avg_memory_mb': sum(memory_values) / len(memory_values),
            'max_memory_mb': max(memory_values),
            'avg_time_minutes': sum(time_values) / len(time_values),
            'total_time_hours': sum(time_values) / 60,
            'avg_gpu_memory_mb': sum(gpu_values) / len(gpu_values) if gpu_values else 0,
            'max_gpu_memory_mb': max(gpu_values) if gpu_values else 0
        }
    
    def shutdown(self):
        """Clean shutdown of resource manager."""
        if self.monitor:
            self.monitor.stop_monitoring()
        
        # Save final metrics
        if self.save_metrics and self.metrics_history:
            metrics_file = Path("final_resource_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump({
                    'metrics_history': self.metrics_history,
                    'resource_summary': self.get_resource_summary()
                }, f, indent=2)
            
            logger.info(f"Resource metrics saved to {metrics_file}")