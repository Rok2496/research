# data/preprocessing/augmentation/cutmix.py
import numpy as np
import torch

class CutMix:
    def __init__(self, beta=1.0):
        self.beta = beta
        
    def apply(self, images, targets):
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)
        
        # Generate random box
        lam = np.random.beta(self.beta, self.beta)
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        
        # Apply CutMix
        images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / 
                   (images.size()[-1] * images.size()[-2]))
        
        mixed_targets = lam * targets + (1 - lam) * targets[index]
        return images, mixed_targets
    
    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2