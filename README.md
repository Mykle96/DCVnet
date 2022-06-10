# DVFnet

This repository contains the source code for our master's thesis at the Norwegian University of
Science and Technology.

The code renderes a pipline for calculating the pose of an object from a singel RGB-picture using their mask and a unit vector fields to approximate each keypoint linked to the respective object.

## Installation

### 1. Virtual Environment

First make sure you have initiated a virtual environment. Anaconda was used for this project and can be initiated by running the following commands:

(Make sure you have installed Anaconda first: [Anaconda installer](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) )

```Bash
conda -V #Check that conda is installed and is in your PATH
conda update conda #Check for latest conda
conda create -n yourenvname python=3.8.5 anaconda #Create your virtual environment - with python 3.8.5
conda activate yourenvname
```

Once your virtual environment is activated, install the required packages by either:

```Bash
pip3 install -r requirements
#or
conda install -r requirements
```

(Make sure you have navigated inside the DVFnet folder)

After this process finishes installing you should have all the necessary packages to run the source code.

## Running the code

To train the network(s), the following code snippet shows the necessary commands and calls:

```python

from torch.utils.data import DataLoader
# Local imports
from engine import Model, VectorModel
from dataLoader import SINTEFDataset


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

if POSE:
    # fetch and format the data
    POSE_TRAIN_DATA, POSE_VAL_DATA = SINTEFDataset(TRAIN_DATA_PATH, pose=POSE, transform=None), SINTEFDataset(
        VAL_DATA_PATH, pose=POSE, transform=None)

    # make batches
    POSE_TRAIN, POSE_VAL = DataLoader(POSE_TRAIN_DATA, batch_size=BATCH_SIZE, shuffle=True), DataLoader(
        POSE_VAL_DATA, batch_size=VAL_BATCH_SIZE, shuffle=False)

    # ------ START TRAINING ------
    # vector
    vectorNetwork = VectorModel(name="DVFnet_v.1")
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
    networks = Model(classes=CLASSES, name="UNET_v.1")
    losses = networks.train(dataset=TRAIN,
                            val_dataset=VAL,
                            epochs=EPOCHS,
                            learning_rate=LR,
                            optimizer=OPTIMIZER)
```

