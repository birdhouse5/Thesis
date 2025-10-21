# merge_hpo_results.py
import json
import argparse
from pathlib import Path


def merge_hpo_results(vae_path: str, ppo_path: str, output_path: str = None):
    """Merge VAE and PPO HPO results into single config file."""
    
    with open(vae_path, 'r') as f:
        vae_data = json.load(f)
    
    with open(ppo_path, 'r') as f:
        ppo_data = json.load(f)
    
    # Merge parameters
    merged_params = {**vae_data['best_params'], **ppo_data['best_params']}
    
    # Create merged result
    merged = {
        "study_name": f"merged_{vae_data['study_name']}_{ppo_data['study_name']}",
        "asset_class": vae_data['asset_class'],
        "reward_type": vae_data['reward_type'],
        "encoder": vae_data['encoder'],
        "source": {
            "vae_hpo": {
                "value": vae_data['best_value'],
                "n_trials": vae_data['n_trials']
            },
            "ppo_hpo": {
                "value": ppo_data['best_value'],
                "n_trials": ppo_data['n_trials']
            }
        },
        "best_params": merged_params,
        "note": "Merged from independent VAE and PPO optimizations"
    }
    
    # Save
    if output_path is None:
        output_path = f"hpo_results/best_params_{merged['asset_class']}_{merged['reward_type']}_merged.json"
    
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print(f"✅ Merged parameters saved to {output_path}")
    print(f"\nMerged parameters ({len(merged_params)} total):")
    for k, v in merged_params.items():
        print(f"  {k}: {v}")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_results", type=str, required=True,
                       help="Path to VAE HPO results JSON")
    parser.add_argument("--ppo_results", type=str, required=True,
                       help="Path to PPO HPO results JSON")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path (optional)")
    args = parser.parse_args()
    
    merge_hpo_results(args.vae_results, args.ppo_results, args.output)