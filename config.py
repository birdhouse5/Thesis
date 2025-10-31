from dataclasses import dataclass
from typing import List, Dict, Optional


# --- 1. High-level experiment spec (only varying factors) ---
@dataclass
class ExperimentConfig:
    seed: int
    asset_class: str       
    encoder: str           
    min_horizon: int = 150
    max_horizon: int = 200
    exp_name: Optional[str] = None
    force_recreate: bool = False
    transaction_cost_rate: Optional[float] = None
    inflation_rate: Optional[float] = None
    n_assets: Optional[int] = None
    concentration_penalty: Optional[bool] = None


# --- 2. Generator for all 60 configs ---
def generate_experiment_configs(num_seeds: int = 10) -> List[ExperimentConfig]:
    assets = ["sp500", "crypto"]
    encoders = ["vae", "none", "hmm"]
    configs = []

    for asset in assets:
        for encoder in encoders:
            for seed in range(num_seeds):
                configs.append(
                    ExperimentConfig(
                        seed=seed,
                        asset_class=asset,
                        encoder=encoder
                    )
                )
    return configs


# --- 3. Translation into full training config ---
@dataclass
class TrainingConfig:
    """Full config used by training loop"""
    seed: int
    exp_name: str
    asset_class: str
    data_path: str
    encoder: str
    disable_vae: bool
    latent_dim: int
    hidden_dim: int
    vae_lr: float
    policy_lr: float
    noise_factor: float
    random_policy: bool
    vae_beta: float
    vae_update_freq: int
    seq_len: int
    episodes_per_task: int
    batch_size: int
    vae_batch_size: int
    ppo_epochs: int
    entropy_coef: float
    joint_loss_lambda: float
    max_episodes: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    val_interval: int
    min_episodes_before_stopping: int
    train_end: str
    val_end: str
    num_assets: int
    device: str
    val_episodes: int
    test_episodes: int
    ppo_clip_ratio: float
    value_loss_coef: float
    max_grad_norm: float
    gae_lambda: float
    discount_factor: float
    min_horizon: int
    max_horizon: int
    eta: float
    ppo_minibatch_size: int
    inflation_rate: float   
    n_assets_limit: Optional[int] = None
    rf_rate: float = 0.02
    transaction_cost_rate: float = 0.000
    force_recreate: bool = False
    reward_type: str = "dsr"
    reward_lookback: int = 20
    vae_num_elbo_terms: int = 8
    min_logstd: float = -3.0
    max_logstd: float = -0.3
    concentration_penalty: bool = False
    concentration_target: float = 0.10  
    concentration_lambda: float = 0.1  
    long_only: bool = False
    
    
    

def experiment_to_training_config(exp: ExperimentConfig) -> TrainingConfig:
    # dataset paths
    data_paths = {
        "sp500": "environments/data/sp500_rl_ready_cleaned.parquet",
        "crypto": "environments/data/crypto_rl_ready_cleaned.parquet"
    }

    # end dates (align with dataset availability)
    if exp.asset_class == "sp500":
        train_end, val_end = "2015-12-31", "2020-12-31"
        eta = 0.01
    else:  
        train_end, val_end = "2020-12-31", "2023-12-31" 
        eta = 0.1

    # encoder handling
    if exp.encoder == "vae":
        disable_vae = False
        latent_dim = 32 
    elif exp.encoder == "hmm":
        disable_vae = True
        latent_dim = 4   
    else:  
        disable_vae = True
        latent_dim = 0 

    default_name=f"{exp.asset_class}_{exp.encoder}_seed{exp.seed}"

    cfg = TrainingConfig(
        seed=exp.seed,
        exp_name=exp.exp_name or default_name,
        asset_class=exp.asset_class,
        data_path=data_paths[exp.asset_class],
        n_assets_limit=exp.n_assets,
        encoder=exp.encoder,
        disable_vae=disable_vae,
        latent_dim=latent_dim,
        hidden_dim=512, 
        vae_lr=0.00004409096982106036,       
        policy_lr=0.00002329493575648219,
        noise_factor=0.05, 
        random_policy=False,
        vae_beta=0.0007435972826570025, 
        vae_update_freq=1,
        seq_len=200,
        episodes_per_task=3,
        batch_size=8192,
        vae_batch_size=1024,
        ppo_epochs=8, 
        entropy_coef=0.001, 
        joint_loss_lambda=1.0,
        max_episodes=9000, #TODO
        early_stopping_patience=10,
        early_stopping_min_delta=0.02,
        val_interval=200,
        min_episodes_before_stopping=1500,
        train_end=train_end,
        val_end=val_end,
        num_assets=30,
        device="cuda",
        val_episodes=50,
        test_episodes=100,
        ppo_clip_ratio=0.19086925122925438, 
        value_loss_coef=1.0, 
        max_grad_norm=1.0, 
        gae_lambda=0.95,
        discount_factor=0.99,
        min_horizon=exp.min_horizon,
        max_horizon=exp.max_horizon,
        eta=0.1, 
        rf_rate=0.02,
        transaction_cost_rate=exp.transaction_cost_rate if exp.transaction_cost_rate is not None else 0.0,
        reward_type=exp.reward_type if hasattr(exp, 'reward_type') else "dsr",
        reward_lookback=39, 
        inflation_rate=exp.inflation_rate if exp.inflation_rate is not None else 0.0,
        ppo_minibatch_size = 128,
        vae_num_elbo_terms = 8,
        min_logstd=-3.0,  
        max_logstd=0.5,
        concentration_penalty=exp.concentration_penalty if exp.concentration_penalty is not None else False,
        
    )

    if hasattr(exp, '_hpo_path'):
        cfg._from_hpo_path = exp._hpo_path
    
    return cfg