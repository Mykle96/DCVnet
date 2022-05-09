import torch
import numpy as np
import math
import warnings
import torch.optim as optim
import torchvision as torchvision
import torch.nn as nn
import torchvision.transforms.functional as TF

# Network for detecting vetor fields on masks to find the keypoints in question.
# Cant be too big as speed is key
# Four layers deep, conv-max -> Dconv -> Dconv -> upconv


class DilatedConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilatedConv, self).__init__()
        self.Dconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel=9,
                      stride=1, padding=1, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.2),
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
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

    def forward(self, x):
        return self.Conv(x)


class DCVnet(torch.nn.Module):
    def __init__(
        self, in_channels=3, out_channels=1, features=[64, 128, 256],
    ):
        super(DCVnet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(Conv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(Conv(feature*2, feature))

        self.bottleneck = DilatedConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
