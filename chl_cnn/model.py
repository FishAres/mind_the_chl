"""3-layer CNN, a PyTorch port of the geo-smart Simple CNN (Keras):
Conv 64 -> ReLU -> BN -> Dropout(0.2) -> Conv 32 -> ReLU -> BN -> Dropout(0.2) -> Conv 1 (linear).
"""
import torch
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self, in_channels, width=(64, 32), dropout=0.2, kernel=3):
        super().__init__()
        layers, c = [], in_channels
        for w in width:
            layers += [nn.Conv2d(c, w, kernel, padding=kernel // 2), nn.ReLU(inplace=True),
                       nn.BatchNorm2d(w), nn.Dropout2d(dropout)]
            c = w
        layers.append(nn.Conv2d(c, 1, kernel, padding=kernel // 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)  # (B, H, W) in ln(CHL) units


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
