# core/models/classification/efficientnet_v2.py
import torch.nn as nn
import timm

class EfficientNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'tf_efficientnetv2_l',
            pretrained=True,
            in_chans=1,  # Set input channels to 1 for grayscale
            num_classes=5  # Number of classes for CheXpert
        )
        
        num_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 5)
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = x.mean(dim=[-2, -1])  # Global average pooling
        x = self.classifier(x)
        return x