# core/schedulers.py
from torch.optim.lr_scheduler import _LRScheduler
import math

class CosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, min_lr):
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                    for base_lr in self.base_lrs]
        
        progress = (self.last_epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs)
        return [self.min_lr + (base_lr - self.min_lr) * 
                (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs]