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


def run_sequential_backtest(datasets, policy, encoder, config, split='test') -> Dict[str, float]:
    """
    Run comprehensive sequential backtest on entire validation/test dataset using rolling context window.
    
    Args:
        datasets: Dictionary with 'train', 'val', 'test' Dataset objects  
        policy: Trained policy network
        encoder: Trained encoder (VAE/HMM/None)
        config: Training configuration
        split: Which dataset split to backtest on ('val' or 'test')
        
    Returns:
        Dictionary with backtest results
    """
    from environments.env import MetaEnv, normalize_with_budget_constraint
    
    device = torch.device(config.device)
    dataset = datasets[split]
    
    print(f"Running sequential backtest on {split} split:")
    print(f"  Dataset period: {dataset.get_split_info()['date_range']}")
    print(f"  Total timesteps: {len(dataset)}")
    print(f"  Context window: {config.seq_len}")
    
    policy.eval()
    if encoder is not None:
        encoder.eval()
    
    # Create environment in sequential mode
    # We need to create a mock dataset for MetaEnv - it expects tensor format
    full_window = dataset.get_window(0, len(dataset))
    mock_dataset = {
        'features': torch.tensor(full_window['features'], dtype=torch.float32),
        'raw_prices': torch.tensor(full_window['raw_prices'], dtype=torch.float32)
    }
    
    env = MetaEnv(
        dataset=mock_dataset, # TODO
        feature_columns=dataset.feature_cols,
        seq_len=config.seq_len,
        min_horizon=config.min_horizon,
        max_horizon=config.max_horizon,
        eta=getattr(config, 'eta', 0.05),
        rf_rate=getattr(config, 'rf_rate', 0.02),
        transaction_cost_rate=getattr(config, 'transaction_cost_rate', 0.001),
        steps_per_year=252 if config.asset_class == 'sp500' else 35040
    )
    
    # Enable sequential mode
    env.set_sequential_mode(True)
    env.config = config  # Ensure config is available for _get_latent_for_step
    env.vae = encoder  # Ensure VAE is available
    
    # Initialize tracking
    daily_returns = []
    daily_weights = []
    daily_capital = []
    portfolio_values = []
    
    # Initialize capital tracking
    initial_capital = 100_000.0
    env.current_capital = initial_capital
    env.initial_capital = initial_capital
    env.capital_history = [initial_capital]
    env.log_returns = []
    env.excess_log_returns = []
    env.alpha = 0.0
    env.beta = 0.0
    env.prev_weights = np.zeros(dataset.num_assets, dtype=np.float32)
    
    with torch.no_grad():
        for t in range(len(dataset) - 1):  # -1 because we need t+1 for returns
            
            # Get current observation
            current_obs = mock_dataset['features'][t].numpy()  # [N, F]
            current_obs_tensor = torch.tensor(current_obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Get latent encoding from rolling context (handles initialization automatically)
            if encoder is None or getattr(config, 'disable_vae', False):
                latent = torch.zeros(1, config.latent_dim, device=device)
            else:
                # Use the environment's method which handles rolling context
                if len(rolling_context) > 0:
                    ctx_obs = torch.stack([ctx['observations'] for ctx in rolling_context]).unsqueeze(0)
                    ctx_acts = torch.stack([ctx['actions'] for ctx in rolling_context]).unsqueeze(0)  
                    ctx_rews = torch.stack([ctx['rewards'] for ctx in rolling_context]).unsqueeze(0).unsqueeze(-1)
                    mu, logvar, _ = encoder.encode(ctx_obs, ctx_acts, ctx_rews)
                    latent = encoder.reparameterize(mu, logvar) # Empty trajectory_context since we use rolling
            
            # Get action from policy
            action, _ = policy.act(current_obs_tensor, latent, deterministic=True)
            action_cpu = action.squeeze(0).detach().cpu().numpy()
            
            # Simulate environment step to get reward and update context
            # We manually compute the reward using the same logic as MetaEnv
            env.current_step = t
            reward, weights, w_cash = env.compute_reward_with_capital(action_cpu)
            
            # Store results
            daily_returns.append(env.excess_log_returns[-1] if env.excess_log_returns else 0.0)
            daily_weights.append(weights.copy())
            daily_capital.append(env.current_capital)
            portfolio_values.append(env.current_capital / initial_capital)
            
            # Progress update
            if t % 100 == 0:
                context_size = env.get_rolling_context_size()
                current_return = (env.current_capital - initial_capital) / initial_capital
                print(f"  Step {t:4d}/{len(dataset)-1}: Return = {current_return:.3%}, Capital = ${env.current_capital:,.0f}, Context = {context_size}")
    
    # Calculate comprehensive metrics
    returns_array = np.array(daily_returns)
    capital_array = np.array(daily_capital)
    
    # Determine annualization factor
    steps_per_year = 252 if config.asset_class == 'sp500' else 35040
    
    # Performance metrics
    total_return = (capital_array[-1] - initial_capital) / initial_capital
    annual_return = np.mean(returns_array) * steps_per_year
    annual_volatility = np.std(returns_array) * np.sqrt(steps_per_year)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    
    # Risk metrics
    negative_returns = returns_array[returns_array < 0]
    downside_volatility = np.std(negative_returns) * np.sqrt(steps_per_year) if len(negative_returns) > 0 else 0
    sortino_ratio = annual_return / downside_volatility if downside_volatility > 0 else 0
    
    # Maximum drawdown
    peak = np.maximum.accumulate(capital_array)
    drawdown = (capital_array - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Additional metrics
    win_rate = np.sum(returns_array > 0) / len(returns_array) if len(returns_array) > 0 else 0
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
        'num_trades': len(returns_array),
        'final_capital': capital_array[-1],
        'initial_capital': initial_capital,
        'avg_turnover': np.mean([np.sum(np.abs(w)) for w in daily_weights]),
        # Detailed data for further analysis
        'daily_returns': returns_array.tolist(),
        'daily_capital': capital_array.tolist(),
        'daily_weights': daily_weights,
        'portfolio_values': portfolio_values
    }
    
    print(f"\nSequential Backtest Results ({split} split):")
    print(f"  Total Return: {total_return:.2%}")
    print(f"  Annual Return: {annual_return:.2%}")
    print(f"  Annual Volatility: {annual_volatility:.2%}")
    print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
    print(f"  Max Drawdown: {max_drawdown:.2%}")
    print(f"  Win Rate: {win_rate:.2%}")
    
    # Reset models to training mode
    policy.train()
    if encoder is not None:
        encoder.train()
    
    return backtest_results