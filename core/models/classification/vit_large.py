import torch
import torch.nn as nn
import timm

class ViTLarge(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_large_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes
        )
        
    def forward(self, x):
        return self.backbone(x)