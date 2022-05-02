import torch
import numpy as np
import tqdm
import trainer
import os
import sys
import pandas
import cython
import warnings
from vectorField import VectorField
from dataLoader import DataLoader
from network import UNET
from vectorField import VectorField

# engine function for training (and validation), evaluation


class Model:

    # Networks
    DEFAULT = 'UNET'

    def __init__(self, model=DEFAULT, classes=None, segmentation=True, pose_estimation=False, pretrained=False, verbose=True):
        # initialize the model class
        # If verbose is selected give more feedback of the process
        self.model = model

        if model == DEFAULT:
            self._model = UNET()

    def train(self, dataset=None, val_dataset=None, epochs=100, learning_rate=0.005, optimizer, loss_fn, momentum=0.9, weight_decay=0.0005, gamma=0.1, lr_step_size=3, scaler=None):

        # Check if the dataset is converted or not, if not initiate, also check for any validation sets.
        assert dataset is not None, "No dataset was received, make sure to input a dataset"
        if not isinstance(dataset, DataLoader):
            dataset = DataLoader(dataset, shuffle=True)

        if val_dataset is not None and not isinstance(val_dataset, DataLoader):
            val_dataset = DataLoader(val_dataset, shuffle=False)

        # initate training parameters and variables
        train_loss = []
        if val_dataset is not None:
            val_loss = []

        # Select optimizer and tune parameters
        # TODO Add the parameters that are required in the optimizer
        assert type(
            self.optimizer) == "", f"Error catched for the optimizer parameter! Expected the input to be of type string, but got {type(optimizer)}."
        # Get parameters that have grad turned on (i.e. parameters that should be trained)
        parameters = [p for p in self._model.parameters() if p.requires_grad]

        if self.optimizer in ["adam", "Adam"]:
            optimizer = torch.optim.Adam(
                parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        elif self.optmizer in ["sdg", "SDG"]:
            optimizer = torch.optim.SDG(
                parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(
                "The optimizer chosen is not added yet.. please use SDG or Adam")

        # LOAD CHECKPOINT

        # TODO make a check on segmentation and if True, make another traning loop for BB
        # ----- TRAINING LOOP BEGINS -----
        print(f"Beginning traning with {network} network.")
        if pose_estimation:
            # TODO fix pose estimation network
            pass
        for epoch in tqdm(range(epochs)):
            print(f'Epoch {epoch + 1} of {epochs}')

            self._model.train()  # training step initiated

            if verbose:
                print('Starting iteration over training dataset')

            # Iterate over the dataset
            for batch_idx, (data, targets) in dataset:
                data = data.to(device=DEVICE)
                targets = targets.float().unsqueeze(1).to(device=DEVICE)
                # generate pose data (VectorField)
                if pose_estimation:
                    vectorfield = VectorField.calculate_vector_field(
                        targets, data)
                # forward
                with torch.cuda.amp.autocast():
                    predictions = model(data)
                    loss = loss_fn(predictions, targets)

                # backward - calculating and updating the gradients of the network
                optimizer.zero_grad()
                # Compute gradients for each parameter based on the current loss calculation
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

    def evaluate(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

    def save(self, file):
        torch.save(self._model.state_dict(), file)
        if verbose:
            print("Model has been saved!")
