from dataclasses import dataclass, field
from typing import List
from pathlib import Path

@dataclass(frozen=True, kw_only=True)
class Config:
    """Simplified configuration with only essential parameters."""
    
    # Experiment identification
    exp_name: str
    seed: int
    
    # Core experimental variables (the ones that actually vary)
    asset_class: str  # "sp500" or "crypto"  
    encoder: str      # "vae", "hmm", or "none"
    
    # Training parameters (reasonable defaults, rarely changed)
    max_episodes: int = 6000
    val_episodes: int = 50
    test_episodes: int = 100
    val_interval: int = 200
    
    # Model architecture (stable defaults)
    latent_dim: int = 512
    hidden_dim: int = 1024
    
    # Training hyperparameters (from HPO, stable)
    policy_lr: float = 0.002
    vae_lr: float = 0.001
    vae_beta: float = 0.013
    batch_size: int = 8192
    ppo_epochs: int = 8
    
    # Environment parameters (computed from asset_class)
    seq_len: int = field(init=False)
    min_horizon: int = field(init=False) 
    max_horizon: int = field(init=False)
    eta: float = field(init=False)
    rf_rate: float = 0.02
    transaction_cost_rate: float = 0.001
    
    # Paths (computed from asset_class)
    data_path: str = field(init=False)
    train_end: str = field(init=False)
    val_end: str = field(init=False)
    
    # Technical settings (rarely changed)
    device: str = "cuda"
    num_assets: int = 30  # Will be updated from actual data
    
    def __post_init__(self):
        """Set computed fields based on asset_class."""
        
        # Data paths and splits
        if self.asset_class == "sp500":
            object.__setattr__(self, 'data_path', "environments/data/sp500_rl_ready_cleaned.parquet")
            object.__setattr__(self, 'train_end', "2015-12-31")
            object.__setattr__(self, 'val_end', "2020-12-31")
            object.__setattr__(self, 'eta', 0.01)
        elif self.asset_class == "crypto":
            object.__setattr__(self, 'data_path', "environments/data/crypto_rl_ready_cleaned.parquet")
            # Crypto dates will be computed from data in prepare_environments()
            object.__setattr__(self, 'train_end', "")  # Will be set later
            object.__setattr__(self, 'val_end', "")    # Will be set later
            object.__setattr__(self, 'eta', 0.1)
        else:
            raise ValueError(f"Unknown asset_class: {self.asset_class}")
            
        # Sequence parameters
        object.__setattr__(self, 'seq_len', 200)
        object.__setattr__(self, 'min_horizon', 150)
        object.__setattr__(self, 'max_horizon', 200)
    
    @property
    def disable_vae(self) -> bool:
        """Whether to disable VAE based on encoder type."""
        return self.encoder in ["hmm", "none"]
    
    @property  
    def vae_update_freq(self) -> int:
        """How often to update VAE."""
        return 5
    
    @property
    def discount_factor(self) -> float:
        """PPO discount factor."""
        return 0.99
        
    @property
    def gae_lambda(self) -> float:
        """GAE lambda parameter."""
        return 0.95
        
    @property
    def ppo_clip_ratio(self) -> float:
        """PPO clipping ratio."""
        return 0.2
        
    @property
    def value_loss_coef(self) -> float:
        """Value loss coefficient."""
        return 0.5
        
    @property
    def entropy_coef(self) -> float:
        """Entropy coefficient."""
        return 0.001
        
    @property
    def max_grad_norm(self) -> float:
        """Gradient clipping norm."""
        return 0.5


def create_experiment_configs(num_seeds: int = 10) -> List[Config]:
    """Create all experiment configurations."""
    configs = []
    
    asset_classes = ["sp500", "crypto"] 
    encoders = ["vae", "none", "hmm"]
    
    for asset_class in asset_classes:
        for encoder in encoders:
            for seed in range(num_seeds):
                config = Config(
                    exp_name=f"{asset_class}_{encoder}_seed{seed}",
                    seed=seed,
                    asset_class=asset_class,
                    encoder=encoder
                )
                configs.append(config)
    
    return configs
