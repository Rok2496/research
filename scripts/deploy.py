# scripts/deploy.py
import os
from core.data import DataManager
from logs.log_manager import LogManager
import mlflow
from dotenv import load_dotenv
import yaml

def deploy_model():
    load_dotenv()
    with open('configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    log_manager = LogManager()
    
    try:
        model_path = 'checkpoints/best_model.pt'
        
        with mlflow.start_run():
            mlflow.log_artifact(model_path)
            mlflow.set_tag("stage", "production")
            
            registered_model = mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/{model_path}",
                "chexpert-model"
            )
            
            mlflow.set_tag("model_version", registered_model.version)
            
    except Exception as e:
        logger = log_manager.setup_deployment_logger()
        logger.error(f"Deployment failed: {e}")
        raise
    finally:
        log_manager.close()

if __name__ == "__main__":
    deploy_model()