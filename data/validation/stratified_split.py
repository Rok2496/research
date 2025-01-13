import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Subset

class StratifiedSplitter:
    def __init__(self, dataset, n_splits=5, test_size=0.2):
        self.dataset = dataset
        self.n_splits = n_splits
        self.test_size = test_size
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    def split(self):
        # Get labels for stratification
        labels = []
        for _, label in self.dataset:
            labels.append(torch.argmax(label).item())
        labels = np.array(labels)

        # Generate indices
        indices = np.arange(len(self.dataset))
        
        # Get train/test split
        train_idx, test_idx = next(self.skf.split(indices, labels))
        
        train_dataset = Subset(self.dataset, train_idx)
        test_dataset = Subset(self.dataset, test_idx)
        
        return train_dataset, test_dataset
