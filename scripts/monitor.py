# scripts/monitor.py
from logs.log_manager import LogManager
from core.metrics import compute_metrics
import mlflow
from dotenv import load_dotenv
import yaml

def monitor_model_performance():
    load_dotenv()
    with open('configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    log_manager = LogManager()
    
    try:
        production_model = mlflow.pyfunc.load_model(
            model_uri=f"models:/chexpert-model/production"
        )
        
        metrics = compute_metrics(production_model, test_loader, config['training']['validation']['metrics'])
        
        with mlflow.start_run():
            mlflow.log_metrics(metrics)
            
    except Exception as e:
        logger = log_manager.setup_deployment_logger()
        logger.error(f"Monitoring failed: {e}")
        raise
    finally:
        log_manager.close()

if __name__ == "__main__":
    monitor_model_performance()