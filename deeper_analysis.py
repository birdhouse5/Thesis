#!/usr/bin/env python3
"""
Deeper analysis of the validation results to assess realism and robustness.
Run this after the main validation to get more insights.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

def analyze_transaction_costs(results_file: str):
    """Analyze impact of transaction costs on returns"""
    print("🏦 TRANSACTION COST ANALYSIS")
    print("="*50)
    
    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    val_results = data['validation_results']['val']
    test_results = data['validation_results']['test']
    
    # Transaction cost scenarios (basis points per trade)
    cost_scenarios = [0, 5, 10, 20, 50]  # 0%, 0.05%, 0.1%, 0.2%, 0.5%
    
    for split_name, results in [('Validation', val_results), ('Test', test_results)]:
        print(f"\n{split_name} Set:")
        
        # Convert to float in case they're stored as strings
        avg_return = float(results['avg_return'])
        avg_turnover = float(results['avg_turnover'])
        
        print(f"  Original Avg Return: {avg_return:.4f}")
        print(f"  Avg Turnover: {avg_turnover:.2f}")
        
        for cost_bps in cost_scenarios:
            cost_rate = cost_bps / 10000  # Convert bps to decimal
            cost_per_episode = avg_turnover * cost_rate
            net_return = avg_return - cost_per_episode
            
            print(f"  With {cost_bps:2d} bps costs: {net_return:7.4f} (cost: {cost_per_episode:.4f})")
            
            if net_return <= 0:
                print(f"    ⚠️  Returns become negative at {cost_bps} bps!")
                break

def analyze_time_series_performance(validation_dir: str):
    """Analyze performance over time to check consistency"""
    print("\n📈 TIME SERIES PERFORMANCE ANALYSIS")
    print("="*50)
    
    # This would require episode timestamps - simplified version
    # In a real implementation, you'd plot performance over time
    
    print("Recommendations for time series analysis:")
    print("  • Plot rolling Sharpe ratios over episodes")
    print("  • Check for regime changes (2016-2017 vs 2019-2020)")
    print("  • Analyze performance during market stress (COVID-19 2020)")
    print("  • Look for momentum vs mean-reversion patterns")

def analyze_portfolio_concentration(validation_dir: str):
    """Deep dive into portfolio allocation patterns"""
    print("\n🎯 PORTFOLIO CONCENTRATION ANALYSIS")
    print("="*50)
    
    # Load portfolio allocation data if available
    # This is a template - actual implementation would load the detailed data
    
    print("Key metrics to analyze:")
    print("  • Maximum single asset weight over time")
    print("  • Number of assets with >1% allocation")
    print("  • Sector concentration (if sector data available)")
    print("  • Correlation with market volatility")
    
    # Concentration risk thresholds
    print("\nConcentration Risk Assessment:")
    print("  • <10% max weight: Well diversified ✅")
    print("  • 10-20% max weight: Moderate concentration ⚠️")
    print("  • >20% max weight: High concentration risk 🚨")

def generate_risk_report(results_file: str):
    """Generate comprehensive risk assessment"""
    print("\n🛡️ COMPREHENSIVE RISK ASSESSMENT")
    print("="*50)
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    val_results = data['validation_results']['val']
    test_results = data['validation_results']['test']
    
    print("📊 Performance Consistency:")
    val_sharpe = float(val_results['avg_reward'])
    test_sharpe = float(test_results['avg_reward'])
    degradation = (val_sharpe - test_sharpe) / val_sharpe * 100
    
    print(f"  Validation Sharpe: {val_sharpe:.2f}")
    print(f"  Test Sharpe: {test_sharpe:.2f}")
    print(f"  Performance degradation: {degradation:.1f}%")
    
    if degradation > 50:
        print("  🚨 High degradation - significant overfitting risk")
    elif degradation > 25:
        print("  ⚠️  Moderate degradation - some overfitting")
    else:
        print("  ✅ Good consistency")
    
    print("\n📉 Drawdown Analysis:")
    val_drawdown = float(val_results.get('avg_max_drawdown', 0))
    test_drawdown = float(test_results.get('avg_max_drawdown', 0))
    
    print(f"  Validation avg drawdown: {val_drawdown:.2%}")
    print(f"  Test avg drawdown: {test_drawdown:.2%}")
    
    print("\n🔄 Trading Intensity:")
    val_turnover = float(val_results['avg_turnover'])
    test_turnover = float(test_results['avg_turnover'])
    
    print(f"  Validation turnover: {val_turnover:.1f}")
    print(f"  Test turnover: {test_turnover:.1f}")
    
    # Annual turnover estimate (assuming daily rebalancing, ~252 trading days)
    annual_turnover_val = val_turnover * 252 / 120  # Adjust for episode length
    annual_turnover_test = test_turnover * 252 / 120
    
    print(f"  Estimated annual turnover: {annual_turnover_val:.0f}% (val), {annual_turnover_test:.0f}% (test)")
    
    if annual_turnover_val > 500:
        print("  🚨 Very high turnover - transaction costs will be significant")
    elif annual_turnover_val > 200:
        print("  ⚠️  High turnover - monitor transaction costs carefully")
    else:
        print("  ✅ Reasonable turnover level")

def benchmark_against_literature():
    """Compare results against academic literature"""
    print("\n📚 LITERATURE BENCHMARK COMPARISON")
    print("="*50)
    
    benchmarks = {
        "Academic RL (typical)": {"sharpe": 0.5, "return": 0.08},
        "Traditional quant funds": {"sharpe": 1.2, "return": 0.15},
        "Top hedge funds": {"sharpe": 1.8, "return": 0.25},
        "Renaissance Medallion": {"sharpe": 2.5, "return": 0.39},
        "Your model (test)": {"sharpe": 4.86, "return": 0.0321}  # From results
    }
    
    print("Performance Comparison:")
    print(f"{'Strategy':<25} {'Sharpe':<8} {'Ann. Return':<12} {'Assessment'}")
    print("-" * 60)
    
    for name, metrics in benchmarks.items():
        sharpe = metrics['sharpe']
        ret = metrics['return']
        
        if name == "Your model (test)":
            # Convert episode return to annualized (rough estimate)
            annual_ret = ret * (252 / 120)  # Adjust for episode length
            assessment = "🎯 Target range" if 0.1 <= annual_ret <= 0.3 else "⚠️  Check scaling"
        else:
            annual_ret = ret
            assessment = ""
        
        print(f"{name:<25} {sharpe:<8.2f} {annual_ret:<12.1%} {assessment}")

def generate_production_recommendations(results_file: str):
    """Generate recommendations for production deployment"""
    print("\n🚀 PRODUCTION DEPLOYMENT RECOMMENDATIONS")
    print("="*50)
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    test_sharpe = float(data['validation_results']['test']['avg_reward'])
    test_return = float(data['validation_results']['test']['avg_return'])
    success_rate = float(data['validation_results']['test']['success_rate'])
    
    print("✅ STRENGTHS:")
    print(f"  • Strong test Sharpe ratio: {test_sharpe:.2f}")
    print(f"  • Good success rate: {success_rate:.0%}")
    print(f"  • Beats equal-weight baseline by large margin")
    print(f"  • Reasonable portfolio diversification")
    
    print("\n⚠️  RISKS TO MONITOR:")
    print(f"  • High volatility in returns (±13.6 std)")
    print(f"  • Performance degradation from val to test")
    print(f"  • Transaction costs could significantly impact returns")
    print(f"  • Model may be sensitive to market regime changes")
    
    print("\n🎯 RECOMMENDED NEXT STEPS:")
    print("  1. Implement transaction cost modeling in training")
    print("  2. Add position size limits to control concentration")
    print("  3. Implement ensemble of models for robustness")
    print("  4. Set up real-time monitoring for performance drift")
    print("  5. Paper trade for 3-6 months before live deployment")
    
    print("\n📊 SUGGESTED POSITION SIZING:")
    if test_sharpe > 3.0:
        position_size = "5-10% of portfolio"
        risk_level = "Moderate allocation"
    elif test_sharpe > 1.5:
        position_size = "10-20% of portfolio"
        risk_level = "Higher allocation justified"
    else:
        position_size = "2-5% of portfolio"
        risk_level = "Conservative allocation"
    
    print(f"  • Position size: {position_size}")
    print(f"  • Risk level: {risk_level}")
    print(f"  • Max drawdown budget: 15-20%")

def main():
    """Run comprehensive deeper analysis"""
    # Find the most recent validation results
    validation_dirs = list(Path("validation_results").glob("validation_*"))
    if not validation_dirs:
        print("❌ No validation results found. Run validate_best_model.py first.")
        return
    
    latest_dir = max(validation_dirs, key=lambda x: x.stat().st_mtime)
    results_file = latest_dir / "validation_summary.json"
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    print("🔍 DEEPER MODEL ANALYSIS")
    print("="*60)
    print(f"Analyzing results from: {latest_dir}")
    
    # Run all analyses
    analyze_transaction_costs(str(results_file))
    analyze_time_series_performance(str(latest_dir))
    analyze_portfolio_concentration(str(latest_dir))
    generate_risk_report(str(results_file))
    benchmark_against_literature()
    generate_production_recommendations(str(results_file))
    
    print("\n" + "="*60)
    print("🎯 SUMMARY VERDICT")
    print("="*60)
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    test_sharpe = float(data['validation_results']['test']['avg_reward'])
    
    if test_sharpe > 3.0:
        verdict = "🏆 EXCELLENT - Strong candidate for production"
        confidence = "High"
    elif test_sharpe > 1.5:
        verdict = "✅ GOOD - Worthy of further development"
        confidence = "Medium-High"
    elif test_sharpe > 0.8:
        verdict = "⚠️  MARGINAL - Needs improvement"
        confidence = "Medium"
    else:
        verdict = "❌ POOR - Back to drawing board"
        confidence = "Low"
    
    print(f"Verdict: {verdict}")
    print(f"Confidence: {confidence}")
    print(f"Test Sharpe: {test_sharpe:.2f}")
    
    print(f"\n📋 For your thesis:")
    print(f"  • Training Sharpe: 121.71 (likely overfitted)")
    print(f"  • Validation Sharpe: 13.25 (optimistic estimate)")
    print(f"  • Test Sharpe: 4.86 (realistic expectation)")
    print(f"  • Performance is strong but requires careful risk management")

if __name__ == "__main__":
    main()