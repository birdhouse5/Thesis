# In main.py
import torch
import logging
from logger_config import setup_experiment_logging
from environments.dataset import Dataset
from environments.env import Environment
from models.vae import VAE
from models.policy import Policy
from algorithms.trainer import Trainer

logger = logging.getLogger(__name__)

class Config:
    """Training configuration"""
    def __init__(self):
        # Training parameters
        self.max_episodes = 1000
        self.episode_length = 60
        self.num_assets = 30
        
        # Model parameters
        self.latent_dim = 64
        self.hidden_dim = 256
        self.vae_lr = 1e-4
        self.policy_lr = 3e-4
        
        # Data
        self.data_path = "environments/data/sp500_rl_ready_cleaned_2.parquet"
        
        # Logging
        self.log_interval = 10
        self.save_interval = 100

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

def main():

    # Setup logging
    exp_logger = setup_experiment_logging("portfolio_varibad")
    
    # Initialize config
    config = Config()
    
    logger.info("Starting VariBAD Portfolio Training")
    logger.info(f"Configuration: episodes={config.max_episodes}, "
               f"latent_dim={config.latent_dim}, vae_lr={config.vae_lr}")
    
    # Initialize dataset and environment
    logger.info("Loading dataset...")
    dataset = Dataset(config.data_path)
    env = Environment(dataset, config.episode_length, config.num_assets)
    
    # Get observation shape from first reset
    initial_obs = env.reset()
    obs_shape = initial_obs.shape  # (30, num_features)
    
    logger.info(f"Environment initialized: obs_shape={obs_shape}")
    
    # Initialize models

    device = torch.device(config.device)

    logger.info("Initializing models...")
    action_dim = config.num_assets * 3  # Fixed size for consistency
    vae = VAE(obs_dim=obs_shape, action_dim=action_dim, latent_dim=config.latent_dim, hidden_dim=config.hidden_dim).to(device)
    policy = Policy(obs_shape=obs_shape, latent_dim=config.latent_dim, 
                   num_assets=config.num_assets, hidden_dim=config.hidden_dim).to(device)
    
    # Initialize trainer
    trainer = Trainer(env, vae, policy, config)
    
    # Training loop
    logger.info("Starting training...")
    best_reward = float('-inf')
    
    try:
        for episode in range(config.max_episodes):
            # Train one episode
            episode_reward, vae_loss, policy_loss = trainer.train_episode()
            
            # Logging
            if episode % config.log_interval == 0:
                logger.info(f"Episode {episode:4d}: reward={episode_reward:8.4f}, "
                           f"vae_loss={vae_loss:6.4f}, policy_loss={policy_loss:6.4f}")
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save({
                    'episode': episode,
                    'vae_state_dict': vae.state_dict(),
                    'policy_state_dict': policy.state_dict(),
                    'best_reward': best_reward,
                    'config': config.__dict__
                }, f"{exp_logger.run_dir}/best_model.pt")
                
                logger.info(f"New best reward: {best_reward:.4f} (saved)")
            
            # Periodic save
            if episode % config.save_interval == 0 and episode > 0:
                torch.save({
                    'episode': episode,
                    'vae_state_dict': vae.state_dict(),
                    'policy_state_dict': policy.state_dict(),
                    'trainer_state': {
                        'episode_count': trainer.episode_count,
                        'total_steps': trainer.total_steps
                    },
                    'config': config.__dict__
                }, f"{exp_logger.run_dir}/checkpoint_ep{episode}.pt")
                
                logger.info(f"Checkpoint saved at episode {episode}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    
    finally:
        # Cleanup
        logger.info("Training completed. Cleaning up...")
        exp_logger.close()
        
        # Final save
        torch.save({
            'episode': trainer.episode_count,
            'vae_state_dict': vae.state_dict(),
            'policy_state_dict': policy.state_dict(),
            'final_reward': episode_reward,
            'config': config.__dict__
        }, f"{exp_logger.run_dir}/final_model.pt")
        
        logger.info(f"Final model saved. Total episodes: {trainer.episode_count}")

if __name__ == "__main__":
    main()