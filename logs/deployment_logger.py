# logs/deployment_logger.py
import logging
from logging.handlers import RotatingFileHandler
import os

class DeploymentLogger:
    def __init__(self, log_dir='logs'):
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger('deployment')
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, 'deployment.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)