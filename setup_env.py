import os
import json
import yaml
from pathlib import Path

def setup_environment():
    training_config = {
        'data': {
            'batch_size': 32,
            'num_workers': 4,
            'pin_memory': True,
            'train_path': 'data/train',
            'val_path': 'data/val'
        },
        'preprocessing': {
            'clahe': {
                'enabled': True,
                'clip_limit': 2.0,
                'tile_grid_size': (8, 8)
            },
            'image_size': [320, 320],
            'augmentation': {
                'enabled': True,
                'elastic': {
                    'enabled': True,
                    'alpha': 1000,
                    'sigma': 50
                },
                'mixup': {
                    'enabled': True,
                    'alpha': 0.2
                },
                'cutmix': {
                    'enabled': True,
                    'beta': 1.0
                }
            }
        },
        'logging': {
            'mlflow': {
                'tracking_uri': 'postgresql://postgres:1234@localhost:5432/mlflow',
                'experiment_name': 'chexpert_training'
            }
        },
        'training': {
            'validation': {
                'metrics': ['accuracy', 'auroc', 'f1_score'],
                'frequency': 1
            }
        }
    }
    
    os.makedirs('configs', exist_ok=True)
    with open('configs/training_config.yaml', 'w') as f:
        yaml.safe_dump(training_config, f, default_flow_style=False)

if __name__ == "__main__":
    setup_environment()