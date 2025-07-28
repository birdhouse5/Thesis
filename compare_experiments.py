#!/usr/bin/env python3
"""
Quick comparison of experiment results
"""

import json
import zipfile
from pathlib import Path
import argparse

def quick_compare(result_files):
    """Quick comparison of experiment results"""
    
    results = []
    
    for file_path in result_files:
        path = Path(file_path)
        
        try:
            if path.suffix == '.zip':
                # Extract from zip
                with zipfile.ZipFile(path, 'r') as zf:
                    with zf.open('experiment_metadata.json') as f:
                        metadata = json.load(f)
                    
                    try:
                        with zf.open('training_stats.json') as f:
                            stats = json.load(f)
                    except:
                        stats = {}
            else:
                # Assume it's a metadata file
                with open(path, 'r') as f:
                    metadata = json.load(f)
                stats = {}
            
            result = {
                'name': metadata.get('experiment_name', path.stem),
                'final_reward': metadata.get('final_performance', {}).get('avg_episode_reward', 'N/A'),
                'parameters': metadata.get('model_parameters', 'N/A'),
                'iterations': metadata.get('total_iterations', 'N/A'),
                'file': str(path)
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Failed to load {path}: {e}")
    
    # Sort by final reward
    valid_results = [r for r in results if r['final_reward'] != 'N/A']
    if valid_results:
        valid_results.sort(key=lambda x: x['final_reward'], reverse=True)
    
    # Print comparison
    print(f"\n📊 Experiment Comparison ({len(results)} experiments)")
    print("="*80)
    print(f"{'Name':<25} {'Final Reward':<15} {'Parameters':<12} {'Iterations':<12}")
    print("-"*80)
    
    for result in valid_results + [r for r in results if r['final_reward'] == 'N/A']:
        reward_str = f"{result['final_reward']:.4f}" if result['final_reward'] != 'N/A' else 'N/A'
        params_str = f"{result['parameters']:,}" if result['parameters'] != 'N/A' else 'N/A'
        
        print(f"{result['name']:<25} {reward_str:<15} {params_str:<12} {result['iterations']:<12}")
    
    if valid_results:
        best = valid_results[0]
        print(f"\n🏆 Best: {best['name']} (reward: {best['final_reward']:.4f})")

def main():
    parser = argparse.ArgumentParser(description="Quick experiment comparison")
    parser.add_argument('files', nargs='+', help='Result files (zips or metadata JSONs)')
    
    args = parser.parse_args()
    
    quick_compare(args.files)

if __name__ == "__main__":
    main()