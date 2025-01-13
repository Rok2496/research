import torch
import torch.nn as nn

class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=5, channels=64):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        
        self.init_size = 32
        self.l1 = nn.Sequential(nn.Linear(latent_dim, channels * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels, channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels, channels//2, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels//2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels//2, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
