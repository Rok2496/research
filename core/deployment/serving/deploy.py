# chexpert_advanced/deployment/serving/deploy.py

import torch
from logs import LogManager
from configs import load_config
from core.models import ModelRegistry

def deploy():
    # Initialize components
    log_manager = LogManager()
    model_registry = ModelRegistry()
    
    try:
        # Load deployment config
        deploy_config = load_config('deployment_config.yaml')
        
        # Setup deployment logger
        deploy_logger = log_manager.setup_deployment_logger()
        
        # Load model
        model = model_registry.load_model('final_model')
        
        # Log deployment metrics
        metrics = {
            'model_load_time': 0.5,
            'gpu_memory_allocated': torch.cuda.memory_allocated(),
            'model_size': model.get_memory_size()
        }
        log_manager.log_metrics(metrics, step=1)
        
        deploy_logger.info("Model deployed successfully")
        
    except Exception as e:
        deploy_logger.error(f"Deployment failed: {str(e)}")
        raise
        
    finally:
        log_manager.close()

if __name__ == '__main__':
    deploy()