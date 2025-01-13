# scripts/setup_logging.py
import os
import json

def setup_logging():
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    os.makedirs('logs/tensorboard', exist_ok=True)
    
    # Create monitoring config
    config = {
        "tensorboard": {
            "log_dir": "tensorboard",
            "update_freq": "epoch",
            "flush_secs": 30
        },
        "deployment": {
            "log_level": "INFO",
            "rotation": "1 day",
            "retention": "30 days"
        },
        "mlflow": {
            "experiment_name": "chexpert_preprocessing",
            "tracking_uri": "postgresql"
        }
    }
    
    # Write config file
    with open('logs/monitoring_config.json', 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == '__main__':
    setup_logging()