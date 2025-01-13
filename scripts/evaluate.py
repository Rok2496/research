# scripts/evaluate.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from logs.log_manager import LogManager
from configs import load_config
from core.data import DataManager
from core.models import ModelRegistry
from core.metrics import compute_metrics
import mlflow
from pathlib import Path

def evaluate():
    base_path = Path(__file__).parent.parent
    config = load_config(base_path / 'configs/training_config.yaml')
    
    log_manager = LogManager()
    model_registry = ModelRegistry()
    data_manager = DataManager()
    
    mlflow.set_tracking_uri(config['logging']['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['logging']['mlflow']['experiment_name'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_registry.get_model('efficientnet_v2').to(device)
    
    checkpoint_path = base_path / 'models/checkpoints/best_model.pth'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
    
    train_loader, val_loader = data_manager.get_dataloaders(
        batch_size=32,
        num_workers=0,
        pin_memory=True
    )
    
    with mlflow.start_run():
        val_metrics = compute_metrics(
            model, 
            val_loader, 
            ['accuracy', 'auroc', 'f1_score']
        )
        
        # Add step parameter for log_metrics
        log_manager.log_metrics(val_metrics, step=0)
        mlflow.log_metrics(val_metrics)
        mlflow.pytorch.log_model(model, "model", registered_model_name="chexpert-model")

if __name__ == '__main__':
    evaluate()