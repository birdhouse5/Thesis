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
    
    # === Original VAE parameters ===
    if 'latent_dim' in params:
        cfg.latent_dim = params['latent_dim']
    if 'hidden_dim' in params:
        cfg.hidden_dim = params['hidden_dim']
    if 'vae_lr' in params:
        cfg.vae_lr = params['vae_lr']
    if 'vae_beta' in params:
        cfg.vae_beta = params['vae_beta']
    
    # === VAE-specific parameters ===
    if 'vae_update_freq' in params:
        cfg.vae_update_freq = params['vae_update_freq']
    if 'vae_num_elbo_terms' in params:
        cfg.vae_num_elbo_terms = params['vae_num_elbo_terms']
    
    # === PPO parameters (original) ===
    if 'policy_lr' in params:
        cfg.policy_lr = params['policy_lr']
    if 'entropy_coef' in params:
        cfg.entropy_coef = params['entropy_coef']
    if 'ppo_clip_ratio' in params:
        cfg.ppo_clip_ratio = params['ppo_clip_ratio']
    
    # === PPO-only parameters ===
    if 'ppo_epochs' in params:
        cfg.ppo_epochs = params['ppo_epochs']
    if 'value_loss_coef' in params:
        cfg.value_loss_coef = params['value_loss_coef']
    if 'max_grad_norm' in params:
        cfg.max_grad_norm = params['max_grad_norm']
    if 'gae_lambda' in params:
        cfg.gae_lambda = params['gae_lambda']
    if 'discount_factor' in params:
        cfg.discount_factor = params['discount_factor']
    if 'ppo_minibatch_size' in params:
        cfg.ppo_minibatch_size = params['ppo_minibatch_size']
    if 'min_logstd' in params:  # ✅ Fixed: was best_params
        cfg.min_logstd = params['min_logstd']
    if 'max_logstd' in params:  # ✅ Fixed: was best_params
        cfg.max_logstd = params['max_logstd']
    
    # === Environment parameters ===
    if 'eta' in params:
        cfg.eta = params['eta']
    if 'reward_lookback' in params:
        cfg.reward_lookback = params['reward_lookback']
    
    # Apply reward type from HPO study
    if 'reward_type' in hpo_data:
        cfg.reward_type = hpo_data['reward_type']
    
    # Handle different JSON structures (single HPO vs merged)
    if 'best_value' in hpo_data:
        # Single HPO result
        best_value = hpo_data['best_value']
        print(f"✅ Loaded HPO parameters from {json_path}")
        print(f"   Best value: {best_value:.4f}")
    elif 'source' in hpo_data:
        # Merged HPO result
        vae_value = hpo_data['source']['vae_hpo']['value']
        ppo_value = hpo_data['source']['ppo_hpo']['value']
        print(f"✅ Loaded merged HPO parameters from {json_path}")
        print(f"   VAE optimization value: {vae_value:.4f}")
        print(f"   PPO optimization value: {ppo_value:.4f}")
    else:
        print(f"✅ Loaded HPO parameters from {json_path}")
    
    print(f"   Applied params: {list(params.keys())}")
    
    return cfg