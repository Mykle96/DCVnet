import torch
import numpy as np
import math
import warnings
import torch.optim as optim
import torchvision as torchvision
import torch.nn as nn

# Network for detecting vetor fields on masks to find the keypoints in question.
# Cant be too big as speed is key
# Three layers deep, conv -> Dconv -> Dconv


class DilatedConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilatedConv, self).__init__()
        self.Dconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel=9,
                      stride=1, padding=1, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.Dconv(x)


class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.Conv(x)


class DCVnet(torch.nn.Module):
    def __init__(
        self, in_channels=3, out_channels=1, features=[64, 128, 256],
    ):
        super(DCVnet, self).__init__()
        self.downs = nn.ModuleList()
