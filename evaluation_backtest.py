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

    # Select dataset split first
    dataset = datasets[split]
    # In evaluation_backtest.py, add at the start of run_sequential_backtest:
    logger.info(f"Dataset type: {type(dataset)}")
    logger.info(f"Dataset attributes: {dir(dataset)}")
    if hasattr(dataset, 'get_window_tensor'):
        logger.info("✓ get_window_tensor method exists")
    else:
        logger.error("✗ get_window_tensor method missing")

    # Print info after dataset is defined
    print(f"Running sequential backtest on {split} split:")
    if hasattr(dataset, 'get_split_info'):
        try:
            print(f"  Dataset period: {dataset.get_split_info().get('date_range', 'unknown')}")
        except Exception:
            print(f"  Dataset period: unknown")
    print(f"  Total timesteps: {len(dataset)}")
    print(f"  Context window: {config.seq_len}")
    
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
        steps_per_year=252 if config.asset_class == 'sp500' else 35040
    )

    backtest_logger = BacktestCSVLogger(
        config.exp_name, config.seed, config.asset_class, 
        config.encoder, dataset.num_assets, config.latent_dim
        )

    # Enable sequential mode
    env.set_sequential_mode(True)
    env.config = config  # Ensure config is available for _get_latent_for_step
    env.vae = encoder  # Ensure VAE is available
    
    env.set_task({
        "sequence": {
            "features": full_window['features'],
            "raw_prices": full_window['raw_prices']
        },
        "task_id": f"{split}_sequential"
    })

    # Initialize tracking
    daily_returns = []
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
    # Initialize previous weights on env device
    env.prev_weights = torch.zeros(dataset.num_assets, dtype=torch.float32, device=env.device)
    
    with torch.no_grad():
        for t in range(len(dataset) - 1):
            # --- Get current observation ---
            current_obs_tensor = full_window['features'][t].to(device).to(torch.float32).unsqueeze(0)

            # --- Build rolling context for VAE ---
            if encoder is None or getattr(config, 'disable_vae', False):
                latent = torch.zeros(1, config.latent_dim, device=device)
            else:
                # Slice the last seq_len steps (or fewer if t < seq_len)
                start = max(0, t - config.seq_len + 1)
                end = t + 1
                ctx_obs = full_window['features'][start:end]           # [L, N, F]
                ctx_acts = torch.zeros((end - start, dataset.num_assets), device=device)  # placeholder
                ctx_rews = torch.zeros((end - start, 1), device=device)                  # placeholder

                ctx_obs = ctx_obs.unsqueeze(0).to(device)   # [1, L, N, F]
                ctx_acts = ctx_acts.unsqueeze(0)            # [1, L, N]
                ctx_rews = ctx_rews.unsqueeze(0)            # [1, L, 1]

                mu, logvar, _ = encoder.encode(ctx_obs, ctx_acts, ctx_rews)
                latent = encoder.reparameterize(mu, logvar)

            # --- Policy step ---
            action, _, _ = policy.act(current_obs_tensor, latent, deterministic=True)

            # --- Environment reward step ---
            env.current_step = t
            reward, weights, w_cash, turnover, cost, equal_weight_return, relative_excess_return = env.compute_reward_with_capital(action.squeeze(0))

            # --- Tracking ---
            excess_return = env.excess_log_returns[-1] if env.excess_log_returns else 0.0
            log_return = env.log_returns[-1] if env.log_returns else 0.0
            weights_np = weights.detach().cpu().numpy()

            # Calculate exposures using standard definitions
            # Ensure proper normalization (defensive programming)
            weights_normalized, w_cash_normalized = normalize_with_budget_constraint(weights)

            # Calculate exposures using properly normalized weights
            long_exp = float(weights_normalized[weights_normalized > 0].sum().item())
            short_exp = float(torch.abs(weights_normalized[weights_normalized < 0]).sum().item()) 
            cash_pos = float(w_cash_normalized)
            net_exp = long_exp - short_exp
            gross_exp = long_exp + short_exp
            backtest_logger.log_step(
                step=t, capital=env.current_capital, log_return=log_return, excess_return=excess_return,
                reward=reward, weights=weights_np, long_exposure=long_exp, short_exposure=short_exp,
                cash_position=cash_pos, net_exposure=net_exp, gross_exposure=gross_exp, 
                turnover=turnover, transaction_cost=cost, latent=latent.squeeze().cpu().numpy()
            )

            # Keep minimal tracking for summary metrics
            daily_returns.append(excess_return)
            daily_capital.append(env.current_capital)
            portfolio_values.append(env.current_capital / initial_capital)

            if t % 100 == 0:
                current_return = (env.current_capital - initial_capital) / initial_capital
                print(f"  Step {t:4d}/{len(dataset)-1}: "
                    f"Return = {current_return:.3%}, "
                    f"Capital = ${env.current_capital:,.0f}")

    
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
        'backtest_sharpe': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'var_95': var_95,
        'num_trades': len(returns_array),
        'final_capital': capital_array[-1],
        'initial_capital': initial_capital,
        'avg_turnover': 0.0,  # Turnover now tracked per-step in CSV
        # Detailed data for further analysis
        'daily_returns': returns_array.tolist(),
        'daily_capital': capital_array.tolist(),
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