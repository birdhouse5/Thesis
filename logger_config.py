# logger_config.py
import logging
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class ExperimentLogger:
    def __init__(self, exp_name, log_dir="logs", level=logging.INFO):
        self.exp_name = exp_name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(log_dir, f"{exp_name}_{timestamp}")
        
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Set up the internal logger
        self.logger = logging.getLogger(f"ExperimentLogger_{exp_name}")
        self.logger.setLevel(level)
        
        # Clear any existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Text logging
        self.setup_text_logger(level)
        
        # TensorBoard
        self.tb_writer = SummaryWriter(log_dir=self.run_dir)
        
        self.logger.info(f"Experiment logging initialized: {self.run_dir}")
    
    def setup_text_logger(self, level):
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_file = os.path.join(self.run_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    # Add standard logging methods
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message):
        """Log critical message"""
        self.logger.critical(message)
    
    def log_scalar(self, tag, value, step):
        """Log scalar to TensorBoard"""
        self.tb_writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag_scalar_dict, step):
        """Log multiple scalars to TensorBoard"""
        for tag, value in tag_scalar_dict.items():
            self.tb_writer.add_scalar(tag, value, step)
    
    def log_hyperparams(self, hparams, metrics=None):
        """Log hyperparameters"""
        self.tb_writer.add_hparams(hparams, metrics or {})
    
    def log_histogram(self, tag, values, step):
        """Log histogram to TensorBoard"""
        self.tb_writer.add_histogram(tag, values, step)
    
    def close(self):
        """Close TensorBoard writer"""
        self.tb_writer.close()

# Global logger instance
experiment_logger = None

def setup_experiment_logging(exp_name, log_dir="logs"):
    """Setup global experiment logger"""
    global experiment_logger
    experiment_logger = ExperimentLogger(exp_name, log_dir)
    return experiment_logger