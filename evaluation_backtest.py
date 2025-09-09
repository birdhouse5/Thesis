import torch
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd


def evaluate(env, policy, encoder, config, num_episodes: int = 50) -> Dict[str, float]:
    """
    Evaluate policy performance on environment.
    
    Args:
        env: MetaEnv instance
        policy: Trained policy network
        encoder: Trained encoder (VAE/HMM/None)
        config: Training configuration
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    device = torch.device(config.device)
    
    episode_rewards = []
    episode_returns = []
    episode_volatilities = []
    episode_sharpe_ratios = []
    portfolio_weights_history = []
    
    policy.eval()
    if encoder is not None:
        encoder.eval()
    
    with torch.no_grad():
        for episode in range(num_episodes):
            # Sample and set task
            task = env.sample_task()
            env.set_task(task)
            
            obs = env.reset()
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            episode_reward = 0
            episode_weights = []
            trajectory_context = {'observations': [], 'actions': [], 'rewards': []}
            done = False
            
            while not done:
                # Get latent encoding
                if encoder is None or getattr(config, 'disable_vae', False):
                    latent = torch.zeros(1, config.latent_dim, device=device)
                elif len(trajectory_context['observations']) == 0:
                    latent = torch.zeros(1, config.latent_dim, device=device)
                else:
                    obs_seq = torch.stack(trajectory_context['observations']).unsqueeze(0)
                    action_seq = torch.stack(trajectory_context['actions']).unsqueeze(0)
                    reward_seq = torch.stack(trajectory_context['rewards']).unsqueeze(0).unsqueeze(-1)
                    
                    if hasattr(encoder, 'encode'):
                        if config.encoder == "hmm":
                            latent = encoder.encode(obs_seq, action_seq, reward_seq)
                        else:  # VAE
                            mu, logvar, _ = encoder.encode(obs_seq, action_seq, reward_seq)
                            latent = encoder.reparameterize(mu, logvar)
                    else:
                        latent = torch.zeros(1, config.latent_dim, device=device)
                
                # Get action from policy
                action, _ = policy.act(obs_tensor, latent, deterministic=True)
                action_cpu = action.squeeze(0).detach().cpu().numpy()
                
                # Take environment step
                next_obs, reward, done, info = env.step(action_cpu)
                
                episode_reward += reward
                episode_weights.append(info['weights'].copy())
                
                # Update trajectory context
                trajectory_context['observations'].append(obs_tensor.squeeze(0).detach())
                trajectory_context['actions'].append(action.squeeze(0).detach())
                trajectory_context['rewards'].append(torch.tensor(reward, device=device))
                
                if not done:
                    obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Calculate episode metrics
            final_capital = env.current_capital
            initial_capital = env.initial_capital
            total_return = (final_capital - initial_capital) / initial_capital
            
            # Calculate volatility from log returns
            if len(env.excess_log_returns) > 1:
                returns_array = np.array(env.excess_log_returns)
                volatility = np.std(returns_array) * np.sqrt(252)  # Annualize
                mean_return = np.mean(returns_array) * 252
                sharpe = mean_return / volatility if volatility > 0 else 0
            else:
                volatility = 0
                sharpe = 0
            
            episode_rewards.append(episode_reward)
            episode_returns.append(total_return)
            episode_volatilities.append(volatility)
            episode_sharpe_ratios.append(sharpe)
            portfolio_weights_history.extend(episode_weights)
    
    # Reset models to training mode
    policy.train()
    if encoder is not None:
        encoder.train()
    
    # Calculate summary statistics
    results = {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'avg_volatility': np.mean(episode_volatilities),
        'avg_sharpe': np.mean(episode_sharpe_ratios),
        'max_return': np.max(episode_returns),
        'min_return': np.min(episode_returns),
        'num_episodes': num_episodes
    }
    
    return results


def run_backtest(env, policy, encoder, config, save_details: bool = True) -> Dict[str, float]:
    """
    Run comprehensive backtest on test environment.
    
    Args:
        env: Test environment
        policy: Trained policy
        encoder: Trained encoder
        config: Configuration
        save_details: Whether to return detailed results
        
    Returns:
        Dictionary with backtest results
    """
    device = torch.device(config.device)
    
    # Track detailed performance
    daily_returns = []
    daily_weights = []
    daily_capital = []
    transaction_costs = []
    
    policy.eval()
    if encoder is not None:
        encoder.eval()
    
    with torch.no_grad():
        # Run single long episode for backtest
        task = env.sample_task()
        env.set_task(task)
        
        obs = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        trajectory_context = {'observations': [], 'actions': [], 'rewards': []}
        done = False
        step = 0
        
        while not done:
            # Get latent encoding
            if encoder is None or getattr(config, 'disable_vae', False):
                latent = torch.zeros(1, config.latent_dim, device=device)
            elif len(trajectory_context['observations']) == 0:
                latent = torch.zeros(1, config.latent_dim, device=device)
            else:
                obs_seq = torch.stack(trajectory_context['observations']).unsqueeze(0)
                action_seq = torch.stack(trajectory_context['actions']).unsqueeze(0)
                reward_seq = torch.stack(trajectory_context['rewards']).unsqueeze(0).unsqueeze(-1)
                
                if hasattr(encoder, 'encode'):
                    if config.encoder == "hmm":
                        latent = encoder.encode(obs_seq, action_seq, reward_seq)
                    else:  # VAE
                        mu, logvar, _ = encoder.encode(obs_seq, action_seq, reward_seq)
                        latent = encoder.reparameterize(mu, logvar)
                else:
                    latent = torch.zeros(1, config.latent_dim, device=device)
            
            # Get action
            action, _ = policy.act(obs_tensor, latent, deterministic=True)
            action_cpu = action.squeeze(0).detach().cpu().numpy()
            
            # Take step
            next_obs, reward, done, info = env.step(action_cpu)
            
            # Record data
            daily_returns.append(info.get('excess_log_return', 0))
            daily_weights.append(info['weights'].copy())
            daily_capital.append(info['capital'])
            
            # Update trajectory context
            trajectory_context['observations'].append(obs_tensor.squeeze(0).detach())
            trajectory_context['actions'].append(action.squeeze(0).detach())
            trajectory_context['rewards'].append(torch.tensor(reward, device=device))
            
            if not done:
                obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            step += 1
    
    # Calculate comprehensive metrics
    returns_array = np.array(daily_returns)
    capital_array = np.array(daily_capital)
    
    # Basic performance metrics
    total_return = (capital_array[-1] - capital_array[0]) / capital_array[0]
    annual_return = np.mean(returns_array) * 252
    annual_volatility = np.std(returns_array) * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    
    # Downside metrics
    negative_returns = returns_array[returns_array < 0]
    downside_volatility = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
    sortino_ratio = annual_return / downside_volatility if downside_volatility > 0 else 0
    
    # Maximum drawdown
    peak = np.maximum.accumulate(capital_array)
    drawdown = (capital_array - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win rate
    win_rate = np.sum(returns_array > 0) / len(returns_array) if len(returns_array) > 0 else 0
    
    # Value at Risk (95% confidence)
    var_95 = np.percentile(returns_array, 5) if len(returns_array) > 0 else 0
    
    backtest_results = {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'var_95': var_95,
        'num_steps': step,
        'final_capital': capital_array[-1],
        'initial_capital': capital_array[0]
    }
    
    # Add detailed data if requested
    if save_details:
        backtest_results.update({
            'daily_returns': returns_array.tolist(),
            'daily_capital': capital_array.tolist(),
            'daily_weights': daily_weights
        })
    
    # Reset models
    policy.train()
    if encoder is not None:
        encoder.train()
    
    return backtest_results