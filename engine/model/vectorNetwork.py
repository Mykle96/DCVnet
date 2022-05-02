import torch
import numpy as np
import math
import warnings

# Network for detecting vetor fields on masks to find the keypoints in question.
# Cant be too big as speed is key


class VectorNetwork(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=1, features=[64, 128, 256],
    ):
