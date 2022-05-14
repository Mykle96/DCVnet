import torch
from torch.utils.data import DataLoader

# Local imports
from engine import Model, VectorModel
from dataLoader import SINTEFDataset
from utils.utils import *


# ----- MODELS -----
# choose which model too use: True --> VectorModel, False --> Segmentation
POSE = True

# ----- HYPERPARAMETERS ------
BATCH_SIZE = 1
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
TRAIN_DATA_PATH = "../data/dataset"
VAL_DATA_PATH = "../data/val_dataset"
TEST_DATA_PATH = ""

if POSE:
    # fetch and format the data
    POSE_TRAIN_DATA, POSE_VAL_DATA = SINTEFDataset(TRAIN_DATA_PATH, pose=POSE, transform=None), SINTEFDataset(
        VAL_DATA_PATH, pose=POSE, transform=None)

    # make batches
    POSE_TRAIN, POSE_VAL = DataLoader(POSE_TRAIN_DATA, batch_size=BATCH_SIZE, shuffle=True), DataLoader(
        POSE_VAL_DATA, batch_size=VAL_BATCH_SIZE, shuffle=False)

    # ------ START TRAINING ------
    # vector
    vectorNetwork = VectorModel()
    vectorLosses = vectorNetwork.train(
        dataset=POSE_TRAIN, val_dataset=POSE_VAL_DATA, epochs=EPOCHS)

else:
    # fetch and format the data
    TRAIN_DATA, VAL_DATA = SINTEFDataset(TRAIN_DATA_PATH, transform=None), SINTEFDataset(
        VAL_DATA_PATH, transform=None)

    # make batches
    TRAIN, VAL = DataLoader(TRAIN_DATA, batch_size=BATCH_SIZE, shuffle=True), DataLoader(
        VAL_DATA, batch_size=VAL_BATCH_SIZE, shuffle=False)

    # ------ START TRAINING ------
    # segment
    networks = Model(classes=CLASSES)
    losses = networks.train(dataset=TRAIN,
                            val_dataset=VAL,
                            epochs=EPOCHS,
                            learning_rate=LR,
                            optimizer=OPTIMIZER)
