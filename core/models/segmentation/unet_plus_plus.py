import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        return x

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.encoder = nn.ModuleList([
            self._make_encoder_block(in_channels, 64),
            self._make_encoder_block(64, 128),
            self._make_encoder_block(128, 256),
            self._make_encoder_block(256, 512)
        ])
        
        self.decoder_blocks = nn.ModuleDict({
            "d0_1": DecoderBlock(128, 64),
            "d1_1": DecoderBlock(256, 128),
            "d2_1": DecoderBlock(512, 256),
            "d0_2": DecoderBlock(192, 64),
            "d1_2": DecoderBlock(384, 128),
            "d0_3": DecoderBlock(256, 64),
        })
        
        self.final = nn.Conv2d(64, num_classes, 1)
        
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        e1 = self.encoder[0](x)
        e2 = self.encoder[1](F.max_pool2d(e1, 2))
        e3 = self.encoder[2](F.max_pool2d(e2, 2))
        e4 = self.encoder[3](F.max_pool2d(e3, 2))

        # Decoder path
        d0_1 = self.decoder_blocks["d0_1"](torch.cat([
            e1, F.interpolate(e2, size=e1.shape[2:])
        ], 1))
        
        d1_1 = self.decoder_blocks["d1_1"](torch.cat([
            e2, F.interpolate(e3, size=e2.shape[2:])
        ], 1))
        
        d0_2 = self.decoder_blocks["d0_2"](torch.cat([
            e1, d0_1, F.interpolate(d1_1, size=e1.shape[2:])
        ], 1))

        return self.final(d0_2)
