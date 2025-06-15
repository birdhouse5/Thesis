"""Debug config structure."""
import yaml
import json

with open('configs/varibad_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Current config structure:")
print(json.dumps(config, indent=2))