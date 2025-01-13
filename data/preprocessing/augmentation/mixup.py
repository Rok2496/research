# data/preprocessing/augmentation/mixup.py
import numpy as np
import torch

class MixUp:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def apply(self, images, targets):
        batch_size = images.size(0)
        weights = np.random.beta(self.alpha, self.alpha, batch_size)
        weights = torch.from_numpy(weights).float().to(images.device)
        
        index = torch.randperm(batch_size).to(images.device)
        mixed_images = weights.view(-1, 1, 1, 1) * images + \
                      (1 - weights.view(-1, 1, 1, 1)) * images[index]
        mixed_targets = weights.view(-1, 1) * targets + \
                       (1 - weights.view(-1, 1)) * targets[index]
        
        return mixed_images, mixed_targets