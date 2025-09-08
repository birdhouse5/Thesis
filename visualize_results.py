def create_ablation_analysis(self, figsize=(8, 6)):
        """Analyze VAE vs no-VAE performance"""
        if self.detailed_results is None:
            print("❌ No detailed results for ablation analysis")
            return
        
        # Try to identify ablation runs (assuming naming convention)
        vae_runs = self.detailed_results[~self.detailed_results['seed'].isin([99])]  # Regular runs
        ablation_runs = self.detailed_results[self.detailed_results['seed'].isin([99])]  # Ablation runs
        
        if len(ablation_runs) == 0:
            print("❌ No ablation runs found (looking for seed 99)")
            return
        
        vae_scores = vae_runs['best_val_sharpe']
        ablation_scores = ablation_runs['best_val_sharpe']
        
        # 1. Performance Comparison
        plt.figure(figsize=figsize)
        data_to_plot = [vae_scores, ablation_scores]
        bp = plt.boxplot(data_to_plot, labels=['VariBAD\n(with VAE)', 'Ablation\n(no VAE)'], 
                        patch_artist=True)
        bp#!/usr/bin/env python3
"""
Comprehensive visualization script for VariBAD portfolio optimization results.
Handles validation results, training dynamics, backtesting, and ablation studies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class VariBADVisualizer:
    """Comprehensive visualization suite for VariBAD experiment results"""
    
    def __init__(self, results_dir: str = "validation_results", output_dir: Optional[str] = None):
        self.results_dir = Path(results_dir)
        
        # Set output directory
        if output_dir is not None:
            self.fig_dir = Path(output_dir)
        else:
            self.fig_dir = self.results_dir / "figures"
        self.fig_dir.mkdir(exist_ok=True)
        
        # Load data
        self.detailed_results = self._load_detailed_results()
        self.statistical_summary = self._load_statistical_summary()
        self.backtest_data = self._load_backtest_data()
        
        print(f"📊 Loaded results from {self.results_dir}")
        if self.detailed_results is not None:
            print(f"   - {len(self.detailed_results)} seed runs")
        if self.backtest_data is not None:
            print(f"   - Backtest data: {len(self.backtest_data)} rows")
    
    def _load_detailed_results(self) -> Optional[pd.DataFrame]:
        """Load individual seed results"""
        csv_path = self.results_dir / "detailed_results.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return None
    
    def _load_statistical_summary(self) -> Optional[Dict]:
        """Load statistical summary"""
        json_path = self.results_dir / "statistical_summary.json"
        if json_path.exists():
            with open(json_path) as f:
                return json.load(f)
        return None
    
    def _load_backtest_data(self) -> Optional[pd.DataFrame]:
        """Load backtest comparison data"""
        csv_path = self.results_dir / "backtest_comparison.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
        return None
    
    def create_validation_overview(self, figsize=(15, 10)):
        """Create comprehensive validation results overview"""
        if self.detailed_results is None:
            print("❌ No detailed results found")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('VariBAD Validation Results Overview', fontsize=16, fontweight='bold')
        
        # 1. Performance Distribution
        val_scores = self.detailed_results['best_val_sharpe']
        axes[0, 0].hist(val_scores, bins=12, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(val_scores.mean(), color='red', linestyle='--', 
                          label=f'Mean: {val_scores.mean():.3f}')
        if self.statistical_summary:
            iqm = self.statistical_summary.get('iqm', val_scores.mean())
            axes[0, 0].axvline(iqm, color='blue', linestyle='-', 
                              label=f'IQM: {iqm:.3f}')
        axes[0, 0].set_xlabel('Validation Sharpe Ratio')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Performance Distribution')
        axes[0, 0].legend()
        
        # 2. Box Plot with Statistical Info
        box_data = [val_scores]
        bp = axes[0, 1].boxplot(box_data, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        axes[0, 1].set_ylabel('Validation Sharpe Ratio')
        axes[0, 1].set_title('Performance Spread')
        axes[0, 1].set_xticklabels(['All Seeds'])
        
        # Add statistical annotations
        q25, q75 = val_scores.quantile([0.25, 0.75])
        axes[0, 1].text(1.15, q75, f'Q3: {q75:.3f}', transform=axes[0, 1].transData)
        axes[0, 1].text(1.15, val_scores.median(), f'Median: {val_scores.median():.3f}', 
                       transform=axes[0, 1].transData)
        axes[0, 1].text(1.15, q25, f'Q1: {q25:.3f}', transform=axes[0, 1].transData)
        
        # 3. Training Efficiency
        axes[0, 2].scatter(self.detailed_results['episodes_trained'], 
                          self.detailed_results['best_val_sharpe'],
                          c=self.detailed_results['early_stopped'].map({True: 'red', False: 'blue'}),
                          alpha=0.6, s=50)
        axes[0, 2].set_xlabel('Episodes Trained')
        axes[0, 2].set_ylabel('Best Validation Sharpe')
        axes[0, 2].set_title('Training Efficiency')
        
        # Add legend for colors
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label='Early Stopped'),
                          Patch(facecolor='blue', label='Full Training')]
        axes[0, 2].legend(handles=legend_elements)
        
        # 4. Early Stopping Analysis
        early_stop_data = self.detailed_results['early_stopped'].value_counts()
        labels = ['Full Training', 'Early Stopped']
        sizes = [early_stop_data.get(False, 0), early_stop_data.get(True, 0)]
        colors = ['lightgreen', 'lightcoral']
        axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Early Stopping Rate')
        
        # 5. Performance vs Training Time
        if 'training_time' in self.detailed_results.columns:
            axes[1, 1].scatter(self.detailed_results['training_time'] / 3600, 
                              self.detailed_results['best_val_sharpe'],
                              alpha=0.6, s=50)
            axes[1, 1].set_xlabel('Training Time (hours)')
            axes[1, 1].set_ylabel('Best Validation Sharpe')
            axes[1, 1].set_title('Performance vs Training Time')
        else:
            axes[1, 1].text(0.5, 0.5, 'Training time\ndata not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Training Time Analysis')
        
        # 6. Statistical Summary
        axes[1, 2].axis('off')
        if self.statistical_summary:
            summary_text = self._format_statistical_summary()
            axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        else:
            axes[1, 2].text(0.5, 0.5, 'Statistical summary\nnot available', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Statistical Summary')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'validation_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _format_statistical_summary(self) -> str:
        """Format statistical summary for display"""
        if not self.statistical_summary:
            return "No summary available"
        
        s = self.statistical_summary
        return f"""Runs: {s.get('num_runs', 'N/A')}
Mean: {s.get('mean', 0):.4f}
IQM:  {s.get('iqm', 0):.4f}
Std:  {s.get('std', 0):.4f}
CI:   [{s.get('iqm_ci_low', 0):.4f}, {s.get('iqm_ci_high', 0):.4f}]
Early Stop: {s.get('early_stop_rate', 0)*100:.1f}%"""
    
    def create_backtest_analysis(self, figsize=(15, 12)):
        """Create comprehensive backtest analysis"""
        if self.backtest_data is None:
            print("❌ No backtest data found")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle('VariBAD vs Benchmarks - Backtest Analysis', fontsize=16, fontweight='bold')
        
        # Prepare data
        df = self.backtest_data.copy()
        
        # 1. Cumulative Wealth Evolution
        for strategy in df['model_name'].unique():
            strategy_data = df[df['model_name'] == strategy].sort_values('date')
            if len(strategy_data) > 0:
                linestyle = '-' if 'VariBAD' in strategy else '--'
                linewidth = 2 if 'VariBAD' in strategy else 1.5
                axes[0, 0].plot(strategy_data['date'], strategy_data['wealth'], 
                               label=strategy, linestyle=linestyle, linewidth=linewidth)
        
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].set_title('Cumulative Wealth Evolution')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Final Performance Comparison
        final_performance = df.groupby('model_name')['wealth'].last().sort_values(ascending=True)
        colors = ['red' if 'VariBAD' in name else 'lightblue' for name in final_performance.index]
        bars = axes[0, 1].barh(range(len(final_performance)), final_performance.values, color=colors)
        axes[0, 1].set_yticks(range(len(final_performance)))
        axes[0, 1].set_yticklabels(final_performance.index)
        axes[0, 1].set_xlabel('Final Portfolio Value ($)')
        axes[0, 1].set_title('Final Performance Ranking')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, final_performance.values)):
            axes[0, 1].text(value + max(final_performance) * 0.01, i, f'${value:,.0f}', 
                           va='center', fontsize=9)
        
        # 3. Returns Distribution
        returns_by_strategy = {}
        for strategy in df['model_name'].unique():
            strategy_data = df[df['model_name'] == strategy]
            returns_by_strategy[strategy] = strategy_data['returns'].values
        
        axes[1, 0].boxplot(returns_by_strategy.values(), labels=returns_by_strategy.keys())
        axes[1, 0].set_ylabel('Daily Returns')
        axes[1, 0].set_title('Return Distribution Comparison')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Risk-Return Scatter
        risk_return_data = []
        for strategy in df['model_name'].unique():
            strategy_data = df[df['model_name'] == strategy]
            returns = strategy_data['returns']
            risk_return_data.append({
                'strategy': strategy,
                'return': returns.mean() * 252,  # Annualized
                'risk': returns.std() * np.sqrt(252),  # Annualized
                'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0
            })
        
        rr_df = pd.DataFrame(risk_return_data)
        colors = ['red' if 'VariBAD' in name else 'blue' for name in rr_df['strategy']]
        scatter = axes[1, 1].scatter(rr_df['risk'], rr_df['return'], 
                                    c=colors, s=100, alpha=0.7)
        
        for i, row in rr_df.iterrows():
            axes[1, 1].annotate(row['strategy'], (row['risk'], row['return']), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        axes[1, 1].set_xlabel('Annualized Risk (Std Dev)')
        axes[1, 1].set_ylabel('Annualized Return')
        axes[1, 1].set_title('Risk-Return Profile')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Rolling Sharpe Ratio
        window = 30  # 30-day rolling window
        for strategy in df['model_name'].unique():
            strategy_data = df[df['model_name'] == strategy].sort_values('date')
            if len(strategy_data) >= window:
                rolling_sharpe = strategy_data['returns'].rolling(window).mean() / strategy_data['returns'].rolling(window).std()
                rolling_sharpe *= np.sqrt(252)  # Annualized
                
                linestyle = '-' if 'VariBAD' in strategy else '--'
                linewidth = 2 if 'VariBAD' in strategy else 1.5
                axes[2, 0].plot(strategy_data['date'].iloc[window-1:], rolling_sharpe.iloc[window-1:], 
                               label=strategy, linestyle=linestyle, linewidth=linewidth)
        
        axes[2, 0].set_xlabel('Date')
        axes[2, 0].set_ylabel('Rolling Sharpe Ratio (30-day)')
        axes[2, 0].set_title('Rolling Risk-Adjusted Performance')
        axes[2, 0].legend()
        axes[2, 0].tick_params(axis='x', rotation=45)
        axes[2, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 6. Performance Metrics Table
        axes[2, 1].axis('off')
        metrics_data = []
        for strategy in df['model_name'].unique():
            strategy_data = df[df['model_name'] == strategy]
            returns = strategy_data['returns']
            
            total_return = (strategy_data['wealth'].iloc[-1] / strategy_data['wealth'].iloc[0] - 1) * 100
            ann_return = returns.mean() * 252 * 100
            ann_vol = returns.std() * np.sqrt(252) * 100
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            max_dd = self._calculate_max_drawdown(strategy_data['wealth'].values)
            
            metrics_data.append([
                strategy[:15] + '...' if len(strategy) > 15 else strategy,
                f"{total_return:.1f}%",
                f"{ann_return:.1f}%", 
                f"{ann_vol:.1f}%",
                f"{sharpe:.2f}",
                f"{max_dd:.1f}%"
            ])
        
        # Create table
        table = axes[2, 1].table(cellText=metrics_data,
                                colLabels=['Strategy', 'Total Ret', 'Ann Ret', 'Ann Vol', 'Sharpe', 'Max DD'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        axes[2, 1].set_title('Performance Metrics Summary')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'backtest_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _calculate_max_drawdown(self, wealth_series: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(wealth_series)
        drawdown = (wealth_series - peak) / peak * 100
        return abs(drawdown.min())
    
    def create_ablation_analysis(self, figsize=(12, 8)):
        """Analyze VAE vs no-VAE performance"""
        if self.detailed_results is None:
            print("❌ No detailed results for ablation analysis")
            return
        
        # Try to identify ablation runs (assuming naming convention)
        vae_runs = self.detailed_results[~self.detailed_results['seed'].isin([99])]  # Regular runs
        ablation_runs = self.detailed_results[self.detailed_results['seed'].isin([99])]  # Ablation runs
        
        if len(ablation_runs) == 0:
            print("❌ No ablation runs found (looking for seed 99)")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Ablation Study: VariBAD vs No-VAE', fontsize=16, fontweight='bold')
        
        # 1. Performance Comparison
        vae_scores = vae_runs['best_val_sharpe']
        ablation_scores = ablation_runs['best_val_sharpe']
        
        data_to_plot = [vae_scores, ablation_scores]
        bp = axes[0, 0].boxplot(data_to_plot, labels=['VariBAD\n(with VAE)', 'Ablation\n(no VAE)'], 
                               patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        axes[0, 0].set_ylabel('Validation Sharpe Ratio')
        axes[0, 0].set_title('Performance Comparison')
        
        # Add statistical test
        if len(vae_scores) > 1 and len(ablation_scores) > 0:
            stat, p_value = stats.mannwhitneyu(vae_scores, ablation_scores, alternative='two-sided')
            axes[0, 0].text(0.5, 0.95, f'Mann-Whitney U test\np-value: {p_value:.4f}', 
                           transform=axes[0, 0].transAxes, ha='center', va='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 2. Distribution Comparison
        axes[0, 1].hist(vae_scores, bins=10, alpha=0.7, label='VariBAD (with VAE)', color='lightblue')
        axes[0, 1].axvline(ablation_scores.iloc[0], color='red', linestyle='--', linewidth=2, 
                          label=f'Ablation (no VAE): {ablation_scores.iloc[0]:.3f}')
        axes[0, 1].set_xlabel('Validation Sharpe Ratio')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Score Distribution')
        axes[0, 1].legend()
        
        # 3. Training Efficiency Comparison
        axes[1, 0].scatter(vae_runs['episodes_trained'], vae_runs['best_val_sharpe'], 
                          alpha=0.6, label='VariBAD (with VAE)', color='blue', s=50)
        axes[1, 0].scatter(ablation_runs['episodes_trained'], ablation_runs['best_val_sharpe'], 
                          alpha=0.8, label='Ablation (no VAE)', color='red', s=100, marker='s')
        axes[1, 0].set_xlabel('Episodes Trained')
        axes[1, 0].set_ylabel('Best Validation Sharpe')
        axes[1, 0].set_title('Training Efficiency')
        axes[1, 0].legend()
        
        # 4. Summary Statistics
        axes[1, 1].axis('off')
        
        vae_stats = f"""VariBAD (with VAE):
Mean: {vae_scores.mean():.4f}
Std:  {vae_scores.std():.4f}
Max:  {vae_scores.max():.4f}
Min:  {vae_scores.min():.4f}
N:    {len(vae_scores)}"""
        
        ablation_stats = f"""Ablation (no VAE):
Score: {ablation_scores.iloc[0]:.4f}
Rank: {(vae_scores > ablation_scores.iloc[0]).sum() + 1}/{len(vae_scores) + 1}"""
        
        axes[1, 1].text(0.1, 0.8, vae_stats, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        axes[1, 1].text(0.1, 0.4, ablation_stats, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))
        
        axes[1, 1].set_title('Statistical Comparison')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'ablation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_full_report(self):
        """Generate complete visualization report"""
        print("🎨 Generating VariBAD Results Visualization Report...")
        print("=" * 60)
        
        # 1. Validation Overview
        print("📊 Creating validation overview...")
        self.create_validation_overview()
        
        # 2. Backtest Analysis
        print("📈 Creating backtest analysis...")
        self.create_backtest_analysis()
        
        # 3. Ablation Study
        print("🔬 Creating ablation analysis...")
        self.create_ablation_analysis()
        
        print(f"\n✅ All visualizations saved to: {self.fig_dir}")
        print(f"   - validation_overview.png")
        print(f"   - backtest_analysis.png") 
        print(f"   - ablation_analysis.png")
        
        # Summary insights
        self._print_insights()
    
    def _print_insights(self):
        """Print key insights from the analysis"""
        print("\n" + "="*60)
        print("📋 KEY INSIGHTS")
        print("="*60)
        
        if self.statistical_summary:
            s = self.statistical_summary
            print(f"🎯 Performance: IQM = {s.get('iqm', 0):.4f} (CI: [{s.get('iqm_ci_low', 0):.4f}, {s.get('iqm_ci_high', 0):.4f}])")
            print(f"📊 Stability: {s.get('num_runs', 0)} runs, std = {s.get('std', 0):.4f}")
            print(f"⏰ Efficiency: {s.get('early_stop_rate', 0)*100:.1f}% early stopping rate")
        
        if self.backtest_data is not None:
            df = self.backtest_data
            varibad_data = df[df['model_name'].str.contains('VariBAD', na=False)]
            if len(varibad_data) > 0:
                final_wealth = varibad_data['wealth'].iloc[-1]
                total_return = (final_wealth / 100000 - 1) * 100
                print(f"💰 Backtest: {total_return:.1f}% total return (${final_wealth:,.0f} final value)")
        
        if self.detailed_results is not None and len(self.detailed_results[self.detailed_results['seed'] == 99]) > 0:
            ablation_score = self.detailed_results[self.detailed_results['seed'] == 99]['best_val_sharpe'].iloc[0]
            regular_mean = self.detailed_results[self.detailed_results['seed'] != 99]['best_val_sharpe'].mean()
            improvement = ((regular_mean - ablation_score) / abs(ablation_score)) * 100
            print(f"🧪 VAE Impact: {improvement:+.1f}% improvement over no-VAE baseline")


def main():
    """Main function to run the visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize VariBAD experiment results')
    parser.add_argument('--results-dir', type=str, default='validation_results',
                       help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save visualization plots (default: results-dir/figures)')
    parser.add_argument('--validation-only', action='store_true',
                       help='Only create validation plots')
    parser.add_argument('--backtest-only', action='store_true', 
                       help='Only create backtest plots')
    parser.add_argument('--ablation-only', action='store_true',
                       help='Only create ablation plots')
    
    args = parser.parse_args()
    
    # Initialize visualizer - NOW CORRECTLY PASSING BOTH ARGUMENTS
    viz = VariBADVisualizer(args.results_dir, args.output_dir)
    
    # Create requested visualizations
    if args.validation_only:
        viz.create_validation_overview()
    elif args.backtest_only:
        viz.create_backtest_analysis()  # Fixed method name
    elif args.ablation_only:
        viz.create_ablation_analysis()
    else:
        viz.create_full_report()


if __name__ == "__main__":
    main()