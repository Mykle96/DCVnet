# packages
import torch
import numpy as np
from tqdm import tqdm
import os
import sys
import pandas
import cython
import warnings
import time
import matplotlib.pyplot as plt
# internal imports
from vectorField import VectorField
# from dataLoader import DataLoader
from torch.utils.data import DataLoader
from model.network import UNET
from vectorField import VectorField
#from DCVnet.visuals.visualization import *


# engine function for training (and validation), evaluation


class Model:

    # Networks
    DEFAULT = 'UNET'
    MASKRCNN = ''

    # TODO Move these to a config file
    ADAM = 'Adam'
    SDG = 'SGD'
    LOSS = ""
    SCALER = torch.cuda.amp.GradScaler()

    def __init__(self, model=DEFAULT, classes=None, segmentation=True, pose_estimation=False, device=None, pretrained=False, verbose=True):
        # initialize the model class
        # If verbose is selected give more feedback of the process
        self.model_name = model
        self.pose_estimation = pose_estimation
        self.verbose = verbose
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        if model == self.DEFAULT:
            self._model = UNET()

        if self.pose_estimation:
            if self.verbose:
                print("Preparing pose estimation pipeline")
            # TODO fix pose estimation network
            pass

    def train(self, dataset, val_dataset=None, epochs=100, learning_rate=0.005, optimizer=SDG, momentum=0.9, weight_decay=0.0005, gamma=0.1, lr_step_size=3, scaler=SCALER):
        loss_fn = torch.nn.BCEWithLogitsLoss()
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
                parameters, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer in ["sgd", "SGD"]:
            print("Optimizer: SGD")
            optimizer = torch.optim.SGD(
                parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(
                "The optimizer chosen is not added yet.. please use SGD or Adam")

        # LOAD CHECKPOINT

        # Set model to the correct device
        self._model.to(device=DEVICE)
        losses = []

        # TODO make a check on segmentation and if True, make another traning loop for BB
        # ----- TRAINING LOOP BEGINS -----
        print(f"Beginning traning with {self.model_name} network.")
        for epoch in tqdm(range(epochs)):
            print("="*50)
            print(f'Epoch {epoch + 1} of {epochs}')

            self._model.train()  # training step initiated

            if self.verbose:
                print('Starting iteration over training dataset')

            iterable = tqdm(dataset, position=0,
                            leave=True) if self.verbose else dataset

            for batch_idx, (data, targets) in enumerate(iterable):
                # TODO See if these data handling functions can be done elsewhere
                data = data.float().permute(0, 3, 1, 2).to(device=DEVICE)
                targets = targets.float().unsqueeze(1).to(device=DEVICE)

                if self.pose_estimation:
                    # generate pose data (VectorField)
                    if self.verbose:
                        print("Generating training data for keypoint localization")
                    keypoints = []  # temporary placeholder
                    vectorfield = VectorField(targets, data, keypoints)
                    trainPoseData = vectorfield.calculate_vector_field(
                        targets, data, keypoints)

                # forward
                with torch.cuda.amp.autocast():
                    predictions = self._model(data)
                    loss = loss_fn(predictions, targets)

                    train_loss.append(loss)
                    total_loss = sum(train_loss)
                    # predKey =  trainPoseData

                # backward - calculating and updating the gradients of the network
                optimizer.zero_grad()
                # Compute gradients for each parameter based on the current loss calculation
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            print(f"Train Loss: {total_loss/len(dataset.dataset)}")

            if self.verbose & epoch % 10 == 0:
                self.show_prediction(data, predictions)

            # ------ VALIDATION LOOP BEGINS -------
            if val_dataset is not None:
                avg_loss = 0

                with torch.no_grad():
                    if self.verbose:
                        print("="*50)
                        print("Starting iteration over validation dataset")

                    iterable = tqdm(val_dataset, position=0,
                                    leave=True) if self.verbose else val_dataset

                    for batch_idx, (images, targets) in enumerate(iterable):
                        # TODO fix the validation loop
                        images = images.float().permute(0, 3, 1, 2).to(device=DEVICE)
                        targets = targets.float().unsqueeze(1).to(device=DEVICE)
                        if self.pose_estimation:
                            # generate pose data (VectorField)
                            if self.verbose:
                                print(
                                    "Generating validation data for keypoint localization")
                            keypoints = []  # temporary placeholder
                            vectorfield = VectorField(
                                targets, images, keypoints)
                            trainPoseData = vectorfield.calculate_vector_field(
                                targets, data, keypoints)

                        with torch.cuda.amp.autocast():
                            predictions = self._model(images)
                            losses = loss_fn(predictions, targets)
                            val_loss.append(losses)
                            total_loss = sum(val_loss)
                            avg_loss += total_loss.item()
                    avg_loss /= len(val_dataset.dataset)

            # If epoch is 10, print a prediction

        return train_loss

    def accuracy(self, image, target, thershold=0.5):
        numCorrect = 0
        diceScore = 0
        numPixels = 0

        self._model.eval()

        with torch.no_grad():
            prediction = torch.sigmoid(self._model(image))
            prediction = (prediction > thershold).float()
            numCorrect += (prediction == target).sum()
            numPixels += torch.numel(prediction)
            diceScore += (2 * (prediction * target).sum()) / (
                (prediction + target).sum() + 1e-8
            )
        print(f"Acc: {numCorrect/numPixels*100:.2f}")
        print(f"Dice Score: {diceScore}")
        self._model.train()

    def evaluate(self, model, pred, target, device):
        # measure the time used
        # Evaluate the model with Dice Score (IoU) and loss

        raise NotImplementedError()

    def show_prediction(self, image, prediction):
        # Show prediction
        #fig, (image,prediction) = plt.subplots(1,2)
        image = image.detach().squeeze(0).permute(1, 2, 0).cpu()
        print("IMAGE: ", image.shape)
        prediction = prediction.detach().squeeze(1).permute(1, 2, 0).cpu()
        fig = plt.figure(figsize=(10, 10))
        img = fig.add_subplot(2, 3, 1)
        img.set_title("Image")
        img.imshow(image/255)

        pred = fig.add_subplot(2, 3, 2)
        pred.set_title("Prediction")
        pred.imshow(prediction)
        plt.show()

        return

    def save(self, file):
        torch.save(self._model.state_dict(), file)
        if self.verbose:
            print("Model has been saved!")


class PoseModel:

    def __init__(self, model, verbose=True):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError
