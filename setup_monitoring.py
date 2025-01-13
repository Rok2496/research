import os
import json
from pathlib import Path

def setup_monitoring():
    # Create directories
    os.makedirs('logs/tensorboard', exist_ok=True)
    os.makedirs('mlruns', exist_ok=True)
    
    # Create monitoring config
    config = {
        "tensorboard": {
            "log_dir": "tensorboard",
            "flush_secs": 30
        },
        "mlflow": {
            "experiment_name": "chexpert_training",
            "tracking_uri": "postgresql://postgres:1234@localhost:5432/mlflow"
        },
        "deployment": {
            "log_level": "INFO"
        }
    }
    
    with open('logs/monitoring_config.json', 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    setup_monitoring()