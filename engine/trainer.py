import torch
from torch.utils.data import DataLoader

# Local imports
from engine import Model
from dataLoader import SINTEFDataset
from utils.utils import *

# ----- HYPERPARAMETERS ------
BATCH_SIZE = 32
VAL_BATCH_SIZE = 15
CLASSES = ["container"]

# ----- DATASET LOADERS ------

TRAIN_DATA_PATH = ""
VAL_DATA_PATH = ""

# fetch and format the data
TRAIN_DATA, VAL_DATA = SINTEFDataset(TRAIN_DATA_PATH, pose=False, transform=None), SINTEFDataset(
    VAL_DATA_PATH, pose=False, transform=None)

# make batches
TRAIN, VAL = DataLoader(TRAIN_DATA, batch_size=BATCH_SIZE, shuffle=True), DataLoader(
    VAL_DATA, batch_size=VAL_BATCH_SIZE, shuffle=True)


# ------ START TRAINING ------

networks = Model(classes=CLASSES, pose_estimation=False)

losses = networks.train(TRAIN, VAL)
