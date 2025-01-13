from torch.utils.data import DataLoader
import torch
from .chexpert_dataset import ChexpertDataset
from .synthetic_dataset import SyntheticDataset

class ChexpertDataLoader:
    def __init__(self, csv_path, image_dir, batch_size=32, num_workers=4):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_loaders(self, train_transform=None, val_transform=None):
        train_dataset = ChexpertDataset(
            self.csv_path + 'train.csv',
            self.image_dir,
            transform=train_transform
        )
        
        val_dataset = ChexpertDataset(
            self.csv_path + 'valid.csv',
            self.image_dir,
            transform=val_transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader, val_loader
