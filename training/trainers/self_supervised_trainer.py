import torch
import torch.nn as nn
from ..losses.contrastive_loss import ContrastiveLoss

class SelfSupervisedTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 temperature=0.07,
                 device='cuda',
                 num_epochs=200):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.criterion = ContrastiveLoss(temperature=temperature)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for (x_i, x_j), _ in train_loader:
            x_i = x_i.to(self.device)
            x_j = x_j.to(self.device)
            
            # Get embeddings
            z_i = self.model(x_i)
            z_j = self.model(x_j)
            
            loss = self.criterion(z_i, z_j)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
        
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for (x_i, x_j), _ in val_loader:
                x_i = x_i.to(self.device)
                x_j = x_j.to(self.device)
                
                z_i = self.model(x_i)
                z_j = self.model(x_j)
                
                loss = self.criterion(z_i, z_j)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
