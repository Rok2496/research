import torch
import torch.nn as nn
import timm
from core.attention import SpatialAttention, ChannelAttention

class DenseNet201(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            'densenet201',
            pretrained=pretrained,
            num_classes=0
        )
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(1920)  # DenseNet201 feature channels
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(1920, num_classes)
        )
        
    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.spatial_attention(x)
        x = self.channel_attention(x)
        return self.classifier(x)