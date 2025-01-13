# models/chexpert_model.py
from core.models import BaseModel
import torch.nn as nn
from core.losses import FocalLoss, ContrastiveLoss

class ChexpertModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.features = nn.Sequential(
            nn.Conv2d(config['model']['input_channels'], 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 78 * 78, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, config['model']['num_classes'])
        )
        
        self.focal_loss = FocalLoss(
            alpha=config['training']['losses']['classification']['focal_loss']['alpha'],
            gamma=config['training']['losses']['classification']['focal_loss']['gamma']
        )
        
        self.contrastive_loss = ContrastiveLoss(
            temperature=config['training']['losses']['self_supervised']['contrastive_loss']['temperature']
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x