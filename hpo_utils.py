import json
from pathlib import Path
from config import TrainingConfig

def load_hpo_params(json_path: str, cfg: TrainingConfig) -> TrainingConfig:
    """
    Load HPO parameters from JSON and apply to config.
    
    Args:
        json_path: Path to JSON file with best parameters
        cfg: Training configuration to update
        
    Returns:
        Updated configuration
    """
    with open(json_path, 'r') as f:
        hpo_data = json.load(f)
    
    params = hpo_data['best_params']
    
    # Apply hyperparameters
    if 'latent_dim' in params:
        cfg.latent_dim = params['latent_dim']
    if 'hidden_dim' in params:
        cfg.hidden_dim = params['hidden_dim']
    if 'vae_lr' in params:
        cfg.vae_lr = params['vae_lr']
    if 'policy_lr' in params:
        cfg.policy_lr = params['policy_lr']
    if 'vae_beta' in params:
        cfg.vae_beta = params['vae_beta']
    if 'entropy_coef' in params:
        cfg.entropy_coef = params['entropy_coef']
    if 'ppo_clip_ratio' in params:
        cfg.ppo_clip_ratio = params['ppo_clip_ratio']
    if 'eta' in params:
        cfg.eta = params['eta']
    if 'reward_lookback' in params:
        cfg.reward_lookback = params['reward_lookback']
    
    # Apply reward type from HPO study
    if 'reward_type' in hpo_data:
        cfg.reward_type = hpo_data['reward_type']
    
    print(f"✅ Loaded HPO parameters from {json_path}")
    print(f"   Best value: {hpo_data['best_value']:.4f}")
    print(f"   Applied params: {params}")
    
    return cfg