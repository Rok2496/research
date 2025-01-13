import torch
import torch.nn as nn
import torchvision.models as models

class SimCLR(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.encoder = models.resnet50(pretrained=True)
        dim_mlp = self.encoder.fc.in_features
        
        # Replace original FC layer with MLP
        self.encoder.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, feature_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

    def get_features(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        x = self.encoder.avgpool(x)
        x = torch.flatten(x, 1)
        return x
