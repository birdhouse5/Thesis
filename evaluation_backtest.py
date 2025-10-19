import torch
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
import logging
logger = logging.getLogger(__name__)
from csv_logger import BacktestCSVLogger
from environments.env import normalize_with_budget_constraint

def evaluate(env, policy, encoder, config, mode, num_episodes: int = 50) -> Dict[str, float]:
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
                else:
                    # NEW: Handle empty context better
                    if len(trajectory_context['observations']) == 0:
                        # Use current observation as minimal context
                        dummy_action = torch.zeros(config.num_assets, device=device)
                        dummy_reward = torch.tensor(0.0, device=device)
                        
                        obs_seq = obs_tensor.unsqueeze(0)  # [1, 1, N, F]
                        action_seq = dummy_action.unsqueeze(0).unsqueeze(0)  # [1, 1, N]
                        reward_seq = dummy_reward.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, 1]
                    else:
                        obs_seq = torch.stack(trajectory_context['observations']).unsqueeze(0)
                        action_seq = torch.stack(trajectory_context['actions']).unsqueeze(0)
                        reward_seq = torch.stack(trajectory_context['rewards']).unsqueeze(0).unsqueeze(-1)
                    
                    try:
                        if hasattr(encoder, 'encode'):
                            if config.encoder == "hmm":
                                latent = encoder.encode(obs_seq, action_seq, reward_seq)
                            else:  # VAE
                                mu, logvar, _ = encoder.encode(obs_seq, action_seq, reward_seq)
                                latent = encoder.reparameterize(mu, logvar)
                        else:
                            latent = torch.zeros(1, config.latent_dim, device=device)
                    except Exception as e:
                        print(f"Encoder failed: {e}")
                        latent = torch.zeros(1, config.latent_dim, device=device)
                
                # Get action from policy
                action, _, _ = policy.act(obs_tensor, latent, deterministic=True)
                
                # Take environment step (tensor input)
                next_obs, reward, done, info = env.step(action.squeeze(0))
                
                episode_reward += reward
                if episode_reward < -100:  # Catch when it goes crazy
                    print(f"EPISODE REWARD EXPLODED: {episode_reward}, last reward: {reward}")
                    print(f"Step: {env.current_step}, done: {done}")
                    break
                weights_np = info['weights'].detach().cpu().numpy().copy() if torch.is_tensor(info['weights']) else np.array(info['weights'], dtype=float)
                episode_weights.append(weights_np)
                
                # Update trajectory context
                trajectory_context['observations'].append(obs_tensor.squeeze(0).detach())
                trajectory_context['actions'].append(action.squeeze(0).detach())
                trajectory_context['rewards'].append(torch.tensor(reward, device=device))
                
                if not done:
                    if torch.is_tensor(next_obs):
                        obs_tensor = next_obs.to(device).to(torch.float32).unsqueeze(0)
                    else:
                        obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Calculate episode metrics
            final_capital = env.current_capital
            initial_capital = env.initial_capital
            total_return = (final_capital - initial_capital) / initial_capital
            
            # Calculate volatility from log returns
            if len(env.excess_log_returns) > 1:
                returns_array = np.array(env.excess_log_returns)
                volatility = np.std(returns_array) * np.sqrt(env.steps_per_year)   # 🔥 FIX
                mean_return = np.mean(returns_array) * env.steps_per_year
                sharpe = mean_return / volatility if volatility > 0 else 0
            else:
                volatility, sharpe = 0, 0

            
            episode_rewards.append(episode_reward)
            #print(f"Final episode reward: {episode_reward}")
            episode_returns.append(total_return)
            episode_volatilities.append(volatility)
            episode_sharpe_ratios.append(sharpe)
            portfolio_weights_history.extend(episode_weights)
    
    # Reset models to training mode
    policy.train()
    if encoder is not None:
        encoder.train()
    
    # Calculate summary statistics
    if mode == "validation":
        results = {
            'validation: avg_reward': np.mean(episode_rewards),
            'validation: std_reward': np.std(episode_rewards),
            'validation: avg_return': np.mean(episode_returns),
            'validation: std_return': np.std(episode_returns),
            'validation: avg_volatility': np.mean(episode_volatilities),
            'validation: avg_episode_sharpe': np.mean(episode_sharpe_ratios),
            'validation: max_return': np.max(episode_returns),
            'validation: min_return': np.min(episode_returns),
            'validation: num_episodes': num_episodes
        }
    else:
        results = {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'avg_volatility': np.mean(episode_volatilities),
            'avg_episode_sharpe': np.mean(episode_sharpe_ratios),
            'max_return': np.max(episode_returns),
            'min_return': np.min(episode_returns),
            'num_episodes': num_episodes
        }
    
    return results


def run_sequential_backtest(datasets, policy, encoder, config, split='test') -> Dict[str, float]:
    """
    Run comprehensive sequential backtest on entire validation/test dataset using rolling context window.
    FIXED: Now maintains actual action/reward history instead of zeros.
    """
    from environments.env import MetaEnv, normalize_with_budget_constraint
    
    device = torch.device(config.device)

    # Select dataset split
    dataset = datasets[split]
    
    logger.info(f"Running sequential backtest on {split} split:")
    if hasattr(dataset, 'get_split_info'):
        try:
            logger.info(f"  Dataset period: {dataset.get_split_info().get('date_range', 'unknown')}")
        except Exception:
            logger.info(f"  Dataset period: unknown")
    logger.info(f"  Total timesteps: {len(dataset)}")
    logger.info(f"  Context window: {config.seq_len}")
    
    policy.eval()
    if encoder is not None:
        encoder.eval()
    full_window = dataset.get_window_tensor(0, len(dataset), device='cpu')
    
    env = MetaEnv(
        dataset=full_window,
        feature_columns=dataset.feature_cols,
        seq_len=config.seq_len,
        min_horizon=config.min_horizon,
        max_horizon=config.max_horizon,
        eta=getattr(config, 'eta', 0.05),
        rf_rate=getattr(config, 'rf_rate', 0.02),
        transaction_cost_rate=getattr(config, 'transaction_cost_rate', 0.001),
        steps_per_year=252 if config.asset_class == 'sp500' else 35040,
        inflation_rate=getattr(config, 'inflation_rate', 0.0),  # Use config value
        reward_type=getattr(config, 'reward_type', 'dsr'),
        reward_lookback=getattr(config, 'reward_lookback', 20),
    )

    backtest_logger = BacktestCSVLogger(
        config.exp_name, config.seed, config.asset_class, 
        config.encoder, dataset.num_assets, config.latent_dim
    )

    # Enable sequential mode
    env.set_sequential_mode(True)
    env.config = config
    env.vae = encoder
    
    env.set_task({
        "sequence": {
            "features": full_window['features'],
            "raw_prices": full_window['raw_prices']
        },
        "task_id": f"{split}_sequential"
    })

    # 🔥 FIX: Initialize history buffers for actual context
    obs_history = []
    action_history = []
    reward_history = []
    
    # Initialize tracking
    daily_returns = []
    daily_capital = []
    portfolio_values = []
    
    initial_capital = 100_000.0
    env.current_capital = initial_capital
    env.initial_capital = initial_capital
    env.capital_history = [initial_capital]
    env.log_returns = []
    env.excess_log_returns = []
    env.alpha = 0.0
    env.beta = 0.0
    env.prev_weights = torch.zeros(dataset.num_assets, dtype=torch.float32, device=env.device)
    
    with torch.no_grad():
        for t in range(len(dataset) - 1):
            # Get current observation
            current_obs_tensor = full_window['features'][t].to(device).to(torch.float32).unsqueeze(0)

            # 🔥 FIX: Build rolling context from actual history
            if encoder is None or getattr(config, 'disable_vae', False):
                latent = torch.zeros(1, config.latent_dim, device=device)
            else:
                if len(obs_history) == 0:
                    # First step: use zero latent
                    latent = torch.zeros(1, config.latent_dim, device=device)
                else:
                    # Use actual history for context
                    start_idx = max(0, len(obs_history) - config.seq_len)
                    ctx_obs = torch.stack(obs_history[start_idx:]).unsqueeze(0).to(device)
                    ctx_acts = torch.stack(action_history[start_idx:]).unsqueeze(0).to(device)
                    ctx_rews = torch.stack(reward_history[start_idx:]).unsqueeze(0).unsqueeze(-1).to(device)

                    try:
                        if config.encoder == "hmm":
                            latent = encoder.encode(ctx_obs, ctx_acts, ctx_rews)
                        else:  # VAE
                            mu, logvar, _ = encoder.encode(ctx_obs, ctx_acts, ctx_rews)
                            latent = encoder.reparameterize(mu, logvar)
                    except Exception as e:
                        logger.warning(f"Encoder failed at step {t}: {e}, using zero latent")
                        latent = torch.zeros(1, config.latent_dim, device=device)

            # Policy step
            action, _, _ = policy.act(current_obs_tensor, latent, deterministic=True)

            # Get normalized weights from environment
            env.current_step = t
            reward, weights, w_cash, turnover, cost, equal_weight_return, relative_excess_return = \
                env.compute_reward_with_capital(action.squeeze(0))

            # 🔥 FIX: Store actual normalized weights in history (not raw actions)
            obs_history.append(current_obs_tensor.squeeze(0).detach().cpu())
            action_history.append(weights.detach().cpu())  # Store NORMALIZED weights
            reward_history.append(torch.tensor(reward, dtype=torch.float32))

            # Rest of tracking code...
            excess_return = env.excess_log_returns[-1] if env.excess_log_returns else 0.0
            log_return = env.log_returns[-1] if env.log_returns else 0.0
            weights_np = weights.detach().cpu().numpy()

            long_exp = float(weights[weights > 0].sum().item())
            short_exp = float(torch.abs(weights[weights < 0]).sum().item())
            cash_pos = float(w_cash)
            net_exp = long_exp - short_exp
            gross_exp = long_exp + short_exp
            
            backtest_logger.log_step(
                step=t, capital=env.current_capital, log_return=log_return, excess_return=excess_return,
                reward=reward, weights=weights_np, long_exposure=long_exp, short_exposure=short_exp,
                cash_position=cash_pos, net_exposure=net_exp, gross_exposure=gross_exp, 
                turnover=turnover, transaction_cost=cost, latent=latent.squeeze().cpu().numpy()
            )

            daily_returns.append(excess_return)
            daily_capital.append(env.current_capital)
            portfolio_values.append(env.current_capital / initial_capital)

            if t % 100 == 0:
                current_return = (env.current_capital - initial_capital) / initial_capital
                logger.info(f"  Step {t:4d}/{len(dataset)-1}: "
                          f"Return = {current_return:.3%}, "
                          f"Capital = ${env.current_capital:,.0f}")

    # Calculate metrics (unchanged)
    returns_array = np.array(daily_returns)
    capital_array = np.array(daily_capital)
    steps_per_year = 252 if config.asset_class == 'sp500' else 35040
    
    total_return = (capital_array[-1] - initial_capital) / initial_capital
    annual_return = np.mean(returns_array) * steps_per_year
    annual_volatility = np.std(returns_array) * np.sqrt(steps_per_year)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    
    negative_returns = returns_array[returns_array < 0]
    downside_volatility = np.std(negative_returns) * np.sqrt(steps_per_year) if len(negative_returns) > 0 else 0
    sortino_ratio = annual_return / downside_volatility if downside_volatility > 0 else 0
    
    peak = np.maximum.accumulate(capital_array)
    drawdown = (capital_array - peak) / peak
    max_drawdown = np.min(drawdown)
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    win_rate = np.sum(returns_array > 0) / len(returns_array) if len(returns_array) > 0 else 0
    var_95 = np.percentile(returns_array, 5) if len(returns_array) > 0 else 0
    
    backtest_results = {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'backtest_sharpe': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'var_95': var_95,
        'num_trades': len(returns_array),
        'final_capital': capital_array[-1],
        'initial_capital': initial_capital,
        'avg_turnover': 0.0,
        'daily_returns': returns_array.tolist(),
        'daily_capital': capital_array.tolist(),
        'portfolio_values': portfolio_values
    }
    
    logger.info(f"\nSequential Backtest Results ({split} split):")
    logger.info(f"  Total Return: {total_return:.2%}")
    logger.info(f"  Annual Return: {annual_return:.2%}")
    logger.info(f"  Annual Volatility: {annual_volatility:.2%}")
    logger.info(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
    logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
    logger.info(f"  Win Rate: {win_rate:.2%}")
    
    policy.train()
    if encoder is not None:
        encoder.train()
    
    return backtest_results