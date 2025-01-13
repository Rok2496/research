import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..losses.focal_loss import FocalLoss
from ..losses.dice_loss import DiceLoss

class SupervisedTrainer:
    def __init__(self, 
                 model,
                 optimizer,
                 device='cuda',
                 num_epochs=100,
                 early_stopping_patience=10):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.patience = early_stopping_patience
        
        # Initialize losses
        self.classification_criterion = FocalLoss()
        self.segmentation_criterion = DiceLoss()
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate losses
            if isinstance(outputs, tuple):
                cls_output, seg_output = outputs
                cls_loss = self.classification_criterion(cls_output, labels)
                seg_loss = self.segmentation_criterion(seg_output, labels)
                loss = cls_loss + seg_loss
            else:
                loss = self.classification_criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                
                if isinstance(outputs, tuple):
                    cls_output, seg_output = outputs
                    cls_loss = self.classification_criterion(cls_output, labels)
                    seg_loss = self.segmentation_criterion(seg_output, labels)
                    loss = cls_loss + seg_loss
                else:
                    loss = self.classification_criterion(outputs, labels)
                
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
