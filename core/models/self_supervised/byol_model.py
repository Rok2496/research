import torch
import torch.nn as nn
import torchvision.models as models

class BYOL(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        
        # Online network
        self.online_encoder = models.resnet50(pretrained=True)
        self.online_projector = self._build_mlp(
            self.online_encoder.fc.in_features,
            feature_dim
        )
        self.online_predictor = self._build_mlp(feature_dim, feature_dim)
        
        # Target network
        self.target_encoder = models.resnet50(pretrained=True)
        self.target_projector = self._build_mlp(
            self.target_encoder.fc.in_features,
            feature_dim
        )
        
        # Disable gradients for target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
            
    def _build_mlp(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_dim)
        )
        
    def forward(self, x1, x2):
        online_proj_1 = self.online_projector(self.online_encoder(x1))
        online_proj_2 = self.online_projector(self.online_encoder(x2))
        
        online_pred_1 = self.online_predictor(online_proj_1)
        online_pred_2 = self.online_predictor(online_proj_2)
        
        with torch.no_grad():
            target_proj_1 = self.target_projector(self.target_encoder(x1))
            target_proj_2 = self.target_projector(self.target_encoder(x2))
            
        return online_pred_1, online_pred_2, target_proj_1.detach(), target_proj_2.detach()
