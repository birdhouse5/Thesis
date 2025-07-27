"""
VariBAD Portfolio Optimization Package
Restructured for clean experimentation
"""

# Import main components for easy access
from .data import load_dataset, create_dataset
from .models import VariBADVAE, PortfolioEnvironment
from .trainer import VariBADTrainer
from .utils import TrajectoryBuffer, get_device, count_parameters

__version__ = "0.2.0"
__all__ = [
    'load_dataset',
    'create_dataset', 
    'VariBADVAE',
    'PortfolioEnvironment',
    'VariBADTrainer',
    'TrajectoryBuffer',
    'get_device',
    'count_parameters'
]