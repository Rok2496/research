# import yaml
# import json
# import os

# def load_config(config_path):
#     """Load configuration from YAML or JSON file"""
#     if config_path.endswith('.yaml') or config_path.endswith('.yml'):
#         with open(config_path, 'r') as f:
#             return yaml.safe_load(f)
#     elif config_path.endswith('.json'):
#         with open(config_path, 'r') as f:
#             return json.load(f)
#     else:
#         raise ValueError(f"Unsupported config file format: {config_path}")
import yaml
import os

def load_config(config_name):
    config_path = os.path.join('configs', config_name)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)