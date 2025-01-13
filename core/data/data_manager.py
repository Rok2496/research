# core/data/data_manager.py
from pathlib import Path
from configs import load_config
from torch.utils.data import DataLoader
from data.datasets.chexpert_dataset import ChexpertDataset

class DataManager:
    def __init__(self):
        base_path = Path(__file__).parent.parent.parent
        self.config = load_config(base_path / 'configs/training_config.yaml')

    def get_dataloaders(self, batch_size=32, num_workers=0, pin_memory=True):
        train_dataset = ChexpertDataset(
            self.config['data']['train_path'],
            self.config,
            mode='train'
        )
        
        val_dataset = ChexpertDataset(
            self.config['data']['val_path'],
            self.config,
            mode='val'
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False
        )
        
        return train_loader, val_loader