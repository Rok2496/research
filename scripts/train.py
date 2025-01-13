# scripts/train.py
import os
import torch
from torch.utils.data import DataLoader
from data.datasets.chexpert_dataset import ChexpertDataset
from models.chexpert_model import ChexpertModel
from core.data import DataManager
from logs.log_manager import LogManager
from core.metrics import compute_metrics
from core.optimizers import RAdam
from core.schedulers import CosineScheduler
import mlflow
from dotenv import load_dotenv
import yaml

def load_config():
    with open('configs/training_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def setup_mlflow(config):
    mlflow.set_tracking_uri(config['logging']['mlflow']['tracking_uri'])
    experiment_name = config['logging']['mlflow']['experiment_name']
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

def main():
    load_dotenv()
    config = load_config()
    
    log_manager = LogManager()
    data_manager = DataManager()
    
    try:
        setup_mlflow(config)
        
        # Setup datasets
        train_dataset = ChexpertDataset(
            config['data']['train_path'],
            config,
            mode='train'
        )
        val_dataset = ChexpertDataset(
            config['data']['val_path'],
            config,
            mode='val'
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory']
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory']
        )
        
        model = ChexpertModel(config)
        optimizer = RAdam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            betas=(config['training']['optimizer']['beta1'],
                  config['training']['optimizer']['beta2']),
            eps=config['training']['optimizer']['eps']
        )
        
        scheduler = CosineScheduler(
            optimizer,
            warmup_epochs=config['training']['scheduler']['warmup_epochs'],
            min_lr=config['training']['scheduler']['min_lr']
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        with mlflow.start_run():
            mlflow.log_params(config)
            
            for epoch in range(config['training']['epochs']):
                model.train()
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = model.focal_loss(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    if batch_idx % 10 == 0:
                        log_manager.log_metrics({
                            'train_loss': loss.item()
                        }, step=epoch * len(train_loader) + batch_idx)
                
                if (epoch + 1) % config['training']['validation']['frequency'] == 0:
                    model.eval()
                    val_metrics = compute_metrics(model, val_loader, config['training']['validation']['metrics'])
                    log_manager.log_metrics(val_metrics, step=epoch)
                    
                    val_loss = val_metrics['val_loss']
                    if val_loss < best_val_loss - config['training']['early_stopping']['min_delta']:
                        best_val_loss = val_loss
                        patience_counter = 0
                        torch.save(model.state_dict(), f"checkpoints/best_model.pt")
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= config['training']['early_stopping']['patience']:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break
                
                scheduler.step()
    
    except Exception as e:
        logger = log_manager.setup_deployment_logger()
        logger.error(f"Training failed: {e}")
        raise
    finally:
        log_manager.close()

if __name__ == "__main__":
    main()