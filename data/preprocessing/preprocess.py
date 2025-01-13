# data/preprocessing/preprocess.py
import os
import torch
import time
from core.data import DataManager
from logs.log_manager import LogManager
import mlflow
from dotenv import load_dotenv

def setup_directories():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('logs/tensorboard', exist_ok=True)
    os.makedirs('mlruns', exist_ok=True)

def setup_mlflow():
    experiment_name = "chexpert_preprocessing"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location="mlruns/chexpert_preprocessing"
        )
        mlflow.set_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_id)
    
    return experiment_id

def main():
    load_dotenv()
    setup_directories()
    
    # Initialize managers
    log_manager = LogManager()
    data_manager = DataManager()
    
    try:
        experiment_id = setup_mlflow()
        current_time = int(time.time() * 1000)
        
        preprocessing_config = {
            'clahe': True,
            'gamma_correction': False,
            'augmentation': {
                'elastic': True,
                'mixup': True,
                'cutmix': False
            },
            'timestamp': current_time
        }
        
        with mlflow.start_run():
            data_manager.preprocess_dataset()
            
            stats = data_manager.get_dataset_stats()
            metrics = {
                'raw_total': stats['raw']['total'],
                'processed_total': stats['processed']['total'],
                'processed_train': stats['processed']['train']
            }
            
            log_manager.log_metrics(metrics, step=0)
            mlflow.log_params(preprocessing_config)
            mlflow.set_tag("project", "chexpert")
    
    except Exception as e:
        logger = log_manager.setup_deployment_logger()
        logger.error(f"Preprocessing failed: {e}")
        raise
    finally:
        log_manager.close()

if __name__ == '__main__':
    main()