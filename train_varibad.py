"""Train VariBAD for portfolio optimization."""

import yaml
import torch
import numpy as np
from datetime import datetime
import os
import time

from src.models.varibad_trader import VariBADTrader
from src.data.data_loader import DataLoader
from src.utils.experiment_logger import create_experiment_logger

def prepare_state(episode_data, current_step):
    """Convert raw data to state representation."""
    # TODO: Implement proper state features
    # For now, return dummy state
    return torch.randn(1, 50)

def prepare_trajectory(trajectory_list):
    """Convert list of transitions to tensor for encoder."""
    if not trajectory_list:
        return None
    
    # Concatenate states, actions, rewards into single tensor
    # For now, create dummy trajectory tensor
    # Shape: (1, sequence_length, feature_dim)
    seq_len = len(trajectory_list)
    feature_dim = 100  # Should match encoder input size
    return torch.randn(1, seq_len, feature_dim)

def calculate_reward(weights, returns):
    """Calculate reward from portfolio weights and returns."""
    # Simple return-based reward
    portfolio_return = (weights * returns).sum()
    return portfolio_return

def train():
    # Load config
    with open('configs/varibad_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create logger
    logger = create_experiment_logger(
        "varibad_trading",
        "Training VariBAD for adaptive portfolio optimization"
    )
    logger.log_config(config, 'full_config')
    
    print("=== VariBAD Training Started ===")
    print(f"Training for {config['training']['num_iterations']} iterations")
    print(f"Batch size: {config['episodes']['batch_size']}")
    
    # Initialize data loader
    print("\nLoading market data...")
    start_time = time.time()
    loader = DataLoader(
        assets=config['data']['assets'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        train_end_date=config['data']['train_end_date']
    )
    print(f"Data loaded in {time.time() - start_time:.1f} seconds")
    
    train_sampler = loader.get_episode_sampler('train', config['episodes']['length'])
    test_sampler = loader.get_episode_sampler('test', config['episodes']['length'])
    
    # Initialize model
    print("\nInitializing model...")
    model = VariBADTrader(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Log setup complete
    logger.log_checkpoint("setup", {"status": "complete"})
    
    # Training loop
    print("\nStarting training loop...")
    print("-" * 50)
    
    for iteration in range(config['training']['num_iterations']):
        iter_start = time.time()
        
        # Sample batch of episodes
        episodes = train_sampler.sample_batch(config['episodes']['batch_size'])
        
        total_loss = 0
        total_return = 0
        
        for ep_idx, episode_data in enumerate(episodes):
            trajectory_list = []
            episode_return = 0
            
            # Run episode
            for t in range(len(episode_data) - 1):
                # Get state
                state = prepare_state(episode_data, t)
                
                # Convert trajectory list to tensor
                trajectory_tensor = prepare_trajectory(trajectory_list)
                
                # Get action from model
                weights, posterior = model(state, trajectory_tensor)
                
                # Calculate reward (next day's return)
                # Fix the iloc warning
                returns = torch.tensor(
                    episode_data['Close'].iloc[t+1].values / episode_data['Close'].iloc[t].values - 1
                ).float()
                reward = calculate_reward(weights.squeeze(), returns)
                
                # Store transition (keep as list for now)
                trajectory_list.append((state, weights, reward))
                episode_return += reward.item()
            
            # TODO: Compute actual losses (ELBO + policy gradient)
            # For now, just optimize for returns
            loss = -episode_return  # Negative because we minimize
            total_loss += loss
            total_return += episode_return
        
        # Update model
        optimizer.zero_grad()
        loss_tensor = torch.tensor(total_loss / len(episodes), requires_grad=True)
        loss_tensor.backward()
        optimizer.step()
        
        # Log progress every 10 iterations
        if iteration % 10 == 0:
            avg_return = total_return / len(episodes)
            iter_time = time.time() - iter_start
            
            # Use log_metrics with a dictionary
            logger.log_metrics({
                'iteration': iteration,
                'train_return': avg_return,
                'time_per_iter': iter_time
            })
            
            print(f"Iter {iteration:4d}/{config['training']['num_iterations']} | "
                  f"Avg Return: {avg_return:7.4f} | "
                  f"Time: {iter_time:.1f}s")
        
        # Periodic evaluation
        if iteration % 100 == 0 and iteration > 0:
            print("\nEvaluating on test data...")
            test_episodes = test_sampler.sample_batch(10)
            test_returns = []
            
            with torch.no_grad():
                for test_ep in test_episodes:
                    ep_return = 0
                    for t in range(len(test_ep) - 1):
                        state = prepare_state(test_ep, t)
                        weights, _ = model(state, None)  # No trajectory for first step
                        returns = torch.tensor(
                            test_ep['Close'].iloc[t+1].values / test_ep['Close'].iloc[t].values - 1
                        ).float()
                        ep_return += calculate_reward(weights.squeeze(), returns).item()
                    test_returns.append(ep_return)
            
            avg_test_return = np.mean(test_returns)
            logger.log_metrics({
                'test_return': avg_test_return,
                'test_iteration': iteration
            })
            print(f"Test Return: {avg_test_return:.4f}")
            print("-" * 50)
        
        # Save checkpoint
        if iteration % config['training']['save_every'] == 0 and iteration > 0:
            checkpoint_path = f"checkpoints/model_iter_{iteration}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'iteration': iteration,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': config
            }, checkpoint_path)
            logger.log_checkpoint("save", {"iteration": iteration, "path": checkpoint_path})
            print(f"Checkpoint saved: {checkpoint_path}")
    
    print("\n=== Training Complete ===")
    logger.log_observation("Training completed successfully")
    logger.finalize()

if __name__ == "__main__":
    train()