import torch
from torch.utils.data import DataLoader

# Local imports
from engine import Model
from dataLoader import SINTEFDataset
from utils.utils import *

# ----- MODELS -----
POSE = False
SEG = True
# ----- HYPERPARAMETERS ------
BATCH_SIZE = 32
VAL_BATCH_SIZE = 15
CLASSES = ["container"]
EPOCHS = 150
LR = 0.001  # 0.0005
OPTIMIZER = "Sgd"
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
GAMMA = 0.1
LR_STEP = 3
# ----- DATASET LOADERS ------
TRAIN_DATA_PATH = "data/dataset"
VAL_DATA_PATH = "data/val_dataset"
TEST_DATA_PATH = ""
# fetch and format the data
TRAIN_DATA, VAL_DATA = SINTEFDataset(TRAIN_DATA_PATH, pose=POSE, transform=None), SINTEFDataset(
    VAL_DATA_PATH, pose=POSE, transform=None)

# make batches
TRAIN, VAL = DataLoader(TRAIN_DATA, batch_size=BATCH_SIZE, shuffle=True), DataLoader(
    VAL_DATA, batch_size=VAL_BATCH_SIZE, shuffle=True)


# ------ START TRAINING ------


networks = Model(classes=CLASSES, segmentation=SEG, pose_estimation=POSE)

losses = networks.train(dataset=TRAIN,
                        val_dataset=VAL,
                        epochs=EPOCHS,
                        learning_rate=LR,
                        optimizer=OPTIMIZER)
