import numpy as np
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Subset

class CrossValidator:
    def __init__(self, dataset, n_splits=5, shuffle=True):
        self.dataset = dataset
        self.n_splits = n_splits
        self.kfold = KFold(n_splits=n_splits, shuffle=shuffle)

    def get_folds(self):
        indices = np.arange(len(self.dataset))
        for train_idx, val_idx in self.kfold.split(indices):
            train_dataset = Subset(self.dataset, train_idx)
            val_dataset = Subset(self.dataset, val_idx)
            yield train_dataset, val_dataset

    def cross_validate(self, model_factory, trainer, metrics):
        fold_scores = []
        for fold, (train_dataset, val_dataset) in enumerate(self.get_folds()):
            model = model_factory()
            fold_score = trainer.train(model, train_dataset, val_dataset)
            fold_scores.append(fold_score)
        return np.mean(fold_scores), np.std(fold_scores)
