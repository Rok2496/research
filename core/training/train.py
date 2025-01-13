import torch
from logs import LogManager
from configs import load_config
from core.models import ModelRegistry
from core.data import DataManager
import argparse
from torch.utils.tensorboard import SummaryWriter
import mlflow
import os
from pathlib import Path

def train(args):
    # Initialize components
    log_manager = LogManager()
    model_registry = ModelRegistry()
    data_manager = DataManager()
    
    # Load configurations
    base_path = Path(__file__).parent.parent.parent
    training_config = load_config(base_path / 'configs/training_config.yaml')
    model_config = load_config(base_path / 'configs/model_config.yaml')
    
    # Setup monitoring
    writer = SummaryWriter(
        log_dir=os.path.join('logs', 'tensorboard'),
        flush_secs=30
    )
    
    # Setup model and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_registry.get_model(model_config['architecture']).to(device)
    
    # Setup optimizer
    optimizer = torch.optim.RAdam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(training_config['training']['optimizer']['beta1'], 
               training_config['training']['optimizer']['beta2']),
        eps=training_config['training']['optimizer']['eps']
    )