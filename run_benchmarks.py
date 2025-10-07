"""
Benchmark baseline strategies on test dataset.
Outputs results in same format as agent results for easy comparison.
"""

import torch
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm

from environments.data import PortfolioDataset
from environments.env import MetaEnv, normalize_with_budget_constraint
from csv_logger import BacktestCSVLogger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkStrategy:
    """Base class for benchmark strategies."""
    
    def __init__(self, num_assets: int, device: str = 'cpu'):
        self.num_assets = num_assets
        self.device = torch.device(device)
    
    def get_action(self, obs: torch.Tensor, step: int) -> torch.Tensor:
        """Return portfolio weights for current step."""
        raise NotImplementedError
    
    def reset(self):
        """Reset strategy state for new episode."""
        pass


class BuyAndHoldStrategy(BenchmarkStrategy):
    """Random initial allocation, never rebalance."""
    
    def __init__(self, num_assets: int, seed: int, device: str = 'cpu'):
        super().__init__(num_assets, device)
        self.seed = seed
        self.initial_weights = None
    
    def reset(self):
        """Generate random initial weights."""
        rng = np.random.RandomState(self.seed)
        # Random weights from uniform [-1, 1]
        raw_weights = rng.uniform(-1, 1, self.num_assets)
        self.initial_weights = torch.tensor(raw_weights, dtype=torch.float32, device=self.device)
    
    def get_action(self, obs: torch.Tensor, step: int) -> torch.Tensor:
        """Return initial weights (never rebalance)."""
        return self.initial_weights


class RandomStrategy(BenchmarkStrategy):
    """Random allocation at each timestep."""
    
    def __init__(self, num_assets: int, seed: int, device: str = 'cpu'):
        super().__init__(num_assets, device)
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def reset(self):
        """Reset RNG for new episode."""
        self.rng = np.random.RandomState(self.seed)
    
    def get_action(self, obs: torch.Tensor, step: int) -> torch.Tensor:
        """Generate random weights each step."""
        raw_weights = self.rng.uniform(-1, 1, self.num_assets)
        return torch.tensor(raw_weights, dtype=torch.float32, device=self.device)


class EqualWeightStrategy(BenchmarkStrategy):
    """Equal weight (1/N) portfolio, set once."""
    
    def __init__(self, num_assets: int, device: str = 'cpu'):
        super().__init__(num_assets, device)
        # Equal positive weights: each asset gets 1/N
        self.weights = torch.ones(num_assets, dtype=torch.float32, device=self.device) / num_assets
    
    def get_action(self, obs: torch.Tensor, step: int) -> torch.Tensor:
        """Return equal weights (never changes)."""
        return self.weights


def run_benchmark_backtest(
    dataset_split,
    strategy: BenchmarkStrategy,
    strategy_name: str,
    asset_class: str,
    seed: int = 0
) -> Dict[str, float]:
    """
    Run backtest for a single benchmark strategy.
    
    Args:
        dataset_split: DatasetSplit object for test data
        strategy: BenchmarkStrategy instance
        strategy_name: Name of strategy for logging
        asset_class: "sp500" or "crypto"
        seed: Random seed (or 0 for deterministic strategies)
    
    Returns:
        Dictionary with backtest results
    """
    device = torch.device('cpu')
    
    # Get full test data as tensors
    full_window = dataset_split.get_window_tensor(0, len(dataset_split), device='cpu')
    
    # Determine steps_per_year
    steps_per_year = 252 if asset_class == 'sp500' else 35040
    
    # Create environment (no transaction costs for benchmarks)
    env = MetaEnv(
        dataset={
            'features': full_window['features'],
            'raw_prices': full_window['raw_prices']
        },
        feature_columns=dataset_split.feature_cols,
        seq_len=200,  # Not used for benchmarks
        min_horizon=150,
        max_horizon=200,
        eta=0.05,
        rf_rate=0.02,
        transaction_cost_rate=0.0,  # No transaction costs
        steps_per_year=steps_per_year,
        inflation_rate=0.0,  # No inflation penalty
        reward_type="dsr",
        reward_lookback=20,
        device='cpu'
    )
    
    # Initialize logging
    exp_name = f"benchmark_{strategy_name}"
    backtest_logger = BacktestCSVLogger(
        exp_name, seed, asset_class,
        "benchmark",  # encoder column
        dataset_split.num_assets,
        0  # latent_dim (not applicable)
    )
    
    # Set task
    env.set_task({
        "sequence": {
            "features": full_window['features'],
            "raw_prices": full_window['raw_prices']
        },
        "task_id": f"test_benchmark_{strategy_name}"
    })
    
    # Reset strategy
    strategy.reset()
    
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
    env.prev_weights = torch.zeros(dataset_split.num_assets, dtype=torch.float32, device=device)
    
    logger.info(f"Running {strategy_name} (seed={seed}) on {asset_class} test set...")
    
    # Run backtest
    for t in tqdm(range(len(dataset_split) - 1), desc=f"{strategy_name} backtest"):
        # Get current observation
        current_obs = full_window['features'][t].to(device).unsqueeze(0)
        
        # Get action from strategy
        raw_action = strategy.get_action(current_obs, t)
        
        # Compute reward (environment handles normalization)
        env.current_step = t
        reward, weights, w_cash, turnover, cost, equal_weight_return, relative_excess_return = \
            env.compute_reward_with_capital(raw_action)
        
        # Track metrics
        excess_return = env.excess_log_returns[-1] if env.excess_log_returns else 0.0
        log_return = env.log_returns[-1] if env.log_returns else 0.0
        weights_np = weights.detach().cpu().numpy()
        
        # Calculate exposures
        weights_normalized, w_cash_normalized = normalize_with_budget_constraint(weights)
        long_exp = float(weights_normalized[weights_normalized > 0].sum().item())
        short_exp = float(torch.abs(weights_normalized[weights_normalized < 0]).sum().item())
        cash_pos = float(w_cash_normalized)
        net_exp = long_exp - short_exp
        gross_exp = long_exp + short_exp
        
        # Log step
        backtest_logger.log_step(
            step=t,
            capital=env.current_capital,
            log_return=log_return,
            excess_return=excess_return,
            reward=reward,
            weights=weights_np,
            long_exposure=long_exp,
            short_exposure=short_exp,
            cash_position=cash_pos,
            net_exposure=net_exp,
            gross_exposure=gross_exp,
            turnover=turnover,
            transaction_cost=cost,
            latent=np.zeros(0)  # No latent for benchmarks
        )
        
        # Track for summary
        daily_returns.append(excess_return)
        daily_capital.append(env.current_capital)
        portfolio_values.append(env.current_capital / initial_capital)
    
    # Calculate summary metrics
    returns_array = np.array(daily_returns)
    capital_array = np.array(daily_capital)
    
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
    
    # Win rate and VaR
    win_rate = np.sum(returns_array > 0) / len(returns_array) if len(returns_array) > 0 else 0
    var_95 = np.percentile(returns_array, 5) if len(returns_array) > 0 else 0
    
    results = {
        'strategy': strategy_name,
        'seed': seed,
        'asset_class': asset_class,
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
        'initial_capital': initial_capital
    }
    
    logger.info(f"  {strategy_name} (seed={seed}) Results:")
    logger.info(f"    Total Return: {total_return:.2%}")
    logger.info(f"    Sharpe Ratio: {sharpe_ratio:.3f}")
    logger.info(f"    Max Drawdown: {max_drawdown:.2%}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run benchmark strategies on test dataset")
    parser.add_argument("--asset_class", type=str, required=True, choices=["sp500", "crypto"],
                       help="Asset class to benchmark")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to dataset (optional, uses default if not provided)")
    args = parser.parse_args()
    
    # Determine data path
    if args.data_path is None:
        args.data_path = f"environments/data/{args.asset_class}_rl_ready_cleaned.parquet"
    
    logger.info(f"Loading {args.asset_class} dataset...")
    
    # Load dataset
    if args.asset_class == "crypto":
        dataset = PortfolioDataset(
            asset_class=args.asset_class,
            data_path=args.data_path,
            proportional=True,
            proportions=(0.7, 0.2, 0.1)
        )
    else:
        dataset = PortfolioDataset(
            asset_class=args.asset_class,
            data_path=args.data_path,
            train_end="2015-12-31",
            val_end="2020-12-31",
            proportional=False
        )
    
    # Get test split
    test_split = dataset.get_split('test')
    num_assets = test_split.num_assets
    
    logger.info(f"Test split: {len(test_split)} days, {num_assets} assets")
    
    all_results = []
    
    # 1. Buy and Hold (5 seeds)
    logger.info("\n=== Buy and Hold Strategy ===")
    for seed in range(5):
        strategy = BuyAndHoldStrategy(num_assets, seed=seed)
        results = run_benchmark_backtest(
            test_split, strategy, "buy_and_hold", args.asset_class, seed=seed
        )
        all_results.append(results)
    
    # 2. Random Allocation (5 seeds)
    logger.info("\n=== Random Allocation Strategy ===")
    for seed in range(5):
        strategy = RandomStrategy(num_assets, seed=seed)
        results = run_benchmark_backtest(
            test_split, strategy, "random_allocation", args.asset_class, seed=seed
        )
        all_results.append(results)
    
    # 3. Equal Weight (deterministic, seed=0)
    logger.info("\n=== Equal Weight Strategy ===")
    strategy = EqualWeightStrategy(num_assets)
    results = run_benchmark_backtest(
        test_split, strategy, "equal_weight", args.asset_class, seed=0
    )
    all_results.append(results)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*80)
    
    # Group by strategy
    strategies = {}
    for r in all_results:
        strat_name = r['strategy']
        if strat_name not in strategies:
            strategies[strat_name] = []
        strategies[strat_name].append(r)
    
    for strat_name, strat_results in strategies.items():
        logger.info(f"\n{strat_name.upper()}:")
        logger.info(f"  Number of runs: {len(strat_results)}")
        
        # Calculate statistics
        returns = [r['total_return'] for r in strat_results]
        sharpes = [r['sharpe_ratio'] for r in strat_results]
        drawdowns = [r['max_drawdown'] for r in strat_results]
        
        logger.info(f"  Total Return: {np.mean(returns):.2%} ± {np.std(returns):.2%}")
        logger.info(f"  Sharpe Ratio: {np.mean(sharpes):.3f} ± {np.std(sharpes):.3f}")
        logger.info(f"  Max Drawdown: {np.mean(drawdowns):.2%} ± {np.std(drawdowns):.2%}")
    
    logger.info("\n" + "="*80)
    logger.info("Benchmark results saved to CSV files in respective experiment_logs directories")
    logger.info("="*80)


if __name__ == "__main__":
    main()