# packages
import torch
import numpy as np
from tqdm import tqdm
import trainer
import os
import sys
import pandas
import cython
import warnings
import time

# internal imports
from vectorField import VectorField
#from dataLoader import DataLoader
from torch.utils.data import DataLoader
from model.network import UNET
from vectorField import VectorField
#from DCVnet.visuals.visualization import *


# engine function for training (and validation), evaluation


class Model:

    # Networks
    DEFAULT = 'UNET'
    MASKRCNN = ''
    ADAM = 'Adam'
    SDG = 'SGD'

    def __init__(self, model=DEFAULT, classes=None, segmentation=True, pose_estimation=False, device=None, pretrained=False, verbose=True):
        # initialize the model class
        # If verbose is selected give more feedback of the process
        self.model = model
        self.model_name = model
        self.pose_estimation = pose_estimation
        self.verbose = verbose
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        if model == self.DEFAULT:
            self._model = UNET()

    def train(self, dataset, val_dataset=None, epochs=100, learning_rate=0.005, optimizer=SDG, loss_fn=None, momentum=0.9, weight_decay=0.0005, gamma=0.1, lr_step_size=3, scaler=None):

        # Check if the dataset is converted or not, if not initiate, also check for any validation sets.
        assert dataset is not None, "No dataset was received, make sure to input a dataset"
        if not isinstance(dataset, DataLoader):
            dataset = DataLoader(dataset, shuffle=True)

        if val_dataset is not None and not isinstance(val_dataset, DataLoader):
            val_dataset = DataLoader(val_dataset, shuffle=False)

        DEVICE = self._device

        # initate training parameters and variables
        train_loss = []
        if val_dataset is not None:
            val_loss = []

        # Select optimizer and tune parameters
        assert type(
            optimizer) == str, f"Error catched for the optimizer parameter! Expected the input to be of type string, but got {type(optimizer)}."
        # Get parameters that have grad turned on (i.e. parameters that should be trained)
        parameters = [p for p in self._model.parameters() if p.requires_grad]

        if optimizer in ["adam", "Adam"]:
            print("Optimizer: Adam")
            optimizer = torch.optim.Adam(
                parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        elif optimizer in ["sgd", "SGD"]:
            print("Optimizer: SGD")
            optimizer = torch.optim.SGD(
                parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(
                "The optimizer chosen is not added yet.. please use SGD or Adam")

        # LOAD CHECKPOINT

        # TODO make a check on segmentation and if True, make another traning loop for BB
        # ----- TRAINING LOOP BEGINS -----
        print(f"Beginning traning with {self.model_name} network.")
        if self.pose_estimation:
            # TODO fix pose estimation network
            pass
        for epoch in tqdm(range(epochs)):
            print(f'Epoch {epoch + 1} of {epochs}')

            self._model.train()  # training step initiated

            if self.verbose:
                print('Starting iteration over training dataset')

            iterable = tqdm(dataset, position=0,
                            leave=True) if self.verbose else dataset

            # Iterate over the dataset
            for batch_idx, (data, targets) in enumerate(iterable):
                data = data.to(device=DEVICE)
                targets = targets.float().unsqueeze(1).to(device=DEVICE)
                # generate pose data (VectorField)
                if self.pose_estimation:
                    keypoints = []  # temporary placeholder
                    vectorfield = VectorField(targets, data, keypoints)
                    trainPoseData = vectorfield.calculate_vector_field(
                        targets, data, keypoints)
                # forward
                with torch.cuda.amp.autocast():
                    predictions = self._model(data)
                    loss = loss_fn(predictions, targets)
                    # predKey =  trainPoseData

                # backward - calculating and updating the gradients of the network
                optimizer.zero_grad()
                # Compute gradients for each parameter based on the current loss calculation
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss.append(loss)

        return train_loss

    def evaluate(self, model, pred, target, device):
        # Evaluate the model with Dice Score (IoU) and loss

        raise NotImplementedError()

    def predict(self):
        # measure the time used
        raise NotImplementedError()

    def save(self, file):
        torch.save(self._model.state_dict(), file)
        if self.verbose:
            print("Model has been saved!")
