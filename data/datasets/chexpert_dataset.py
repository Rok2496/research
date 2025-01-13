# data/datasets/chexpert_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class ChexpertDataset(Dataset):
    def __init__(self, data_path, config, mode='train', transform=None):
        super().__init__()
        self.data_path = Path(data_path)
        self.mode = mode
        self.config = config
        self.transform = transform
        self.samples = self._generate_dummy_data()

    def _generate_dummy_data(self):
        samples = []
        num_samples = 100
        for i in range(num_samples):
            image = np.random.randint(0, 255, (320, 320), dtype=np.uint8)
            label = np.random.randint(0, 2, 5).astype(np.float32)
            samples.append((image, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, label = self.samples[idx]
        image = torch.from_numpy(image).float().unsqueeze(0)
        label = torch.from_numpy(label).float()
        
        if self.transform:
            image = self.transform(image)
        
        return image, label