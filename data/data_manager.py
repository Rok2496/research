# data/preprocessing/preprocess.py
import os
import torch
from data.data_manager import DataManager
from logs.log_manager import LogManager
import mlflow
from dotenv import load_dotenv

def setup_directories():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('logs/tensorboard', exist_ok=True)
    os.makedirs('mlruns', exist_ok=True)

def main():
    # Ensure logging setup
    if not os.path.exists('logs/monitoring_config.json'):
        from scripts.setup_logging import setup_logging
        setup_logging()
    
    setup_directories()
    
    # Initialize managers
    log_manager = LogManager()
    data_manager = DataManager()
    
    try:
        preprocessing_config = {
            'clahe': True,
            'gamma_correction': False,
            'augmentation': {
                'elastic': True,
                'mixup': True,
                'cutmix': False
            }
        }
        
        with mlflow.start_run():
            data_manager.preprocess_dataset()
            
            stats = data_manager.get_dataset_stats()
            log_manager.log_metrics({
                'raw_total': stats['raw']['total'],
                'processed_total': stats['processed']['total'],
                'processed_train': stats['processed']['train']
            }, step=0)
            
            mlflow.log_params(preprocessing_config)
    
    except Exception as e:
        logger = log_manager.setup_deployment_logger()
        logger.error(f"Preprocessing failed: {e}")
        raise
    finally:
        log_manager.close()

if __name__ == '__main__':
    main()