# packages
from multiprocessing.sharedctypes import Value
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
import cv2
import math as m
import random
from torch.utils.data import DataLoader

# internal imports
from vectorField import VectorField
# from dataLoader import DataLoader
from model.network import UNET
from model.vectorNetwork import DCVnet
from vectorField import VectorField
from utils.utils import *
# from DCVnet.visuals.visualization import *


# engine function for training (and validation), evaluation


class Model:

    # Networks
    DEFAULT = 'UNET'
    MASKRCNN = ''

    # TODO Move these to a config file
    ADAM = 'Adam'
    SGD = 'SGD'
    LOSS = ""
    SCALER = torch.cuda.amp.GradScaler()

    def __init__(self, model=DEFAULT, classes=None, segmentation=True, pose_estimation=False, save_images=False, verbose=True):
        # initialize the model class
        # If verbose is selected give more feedback of the process
        self.model_name = model
        self.pose_estimation = pose_estimation
        self.verbose = verbose
        self.save_images = save_images
        self.classes = classes
        self.numClasses = len(classes)
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        if model == self.DEFAULT:
            self._model = UNET()

        if self.pose_estimation:
            if self.verbose:
                print("Preparing pose estimation pipeline")
            # TODO fix pose estimation network
            poseNetwork = PoseModel(device=self._device)
            self.poseNetwork = poseNetwork

    def train(self, dataset, val_dataset=None, epochs=150, learning_rate=0.005, optimizer=SGD, loss_fn=None, momentum=0.9, weight_decay=0.0005, gamma=0.1, lr_step_size=3, scaler=SCALER):

        # Check if the dataset is converted or not, if not initiate, also check for any validation sets.
        assert dataset is not None, "No dataset was received, make sure to input a dataset"
        if not isinstance(dataset, DataLoader):
            dataset = DataLoader(dataset, shuffle=True)

        if val_dataset is not None and not isinstance(val_dataset, DataLoader):
            val_dataset = DataLoader(val_dataset, shuffle=True)

        DEVICE = self._device
        # initate training parameters and variables
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
                f"The optimizer chosen: {optimizer}, is either not added yet or invalid.. please use SGD or Adam")
        # Initialize the loss function
        if loss_fn is None:
            if self.numClasses > 1:
                loss_fn = torch.nn.CrossEntropyLoss()
                # If Cross entropy is used, add Sigmoid on the pred before loss is calculated
            if self.numClasses == 1:
                loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            loss_fn = loss_fn

        # LOAD CHECKPOINT

        # Set model to the correct device
        self._model.to(device=DEVICE)
        train_loss = []
        epoch_losses = []
        losses = []

        # TODO make a check on segmentation and if True, make another traning loop for BB
        # ----- TRAINING LOOP BEGINS -----
        print(f"Beginning traning with {self.model_name} network.")
        for epoch in tqdm(range(epochs)):

            print(f'Epoch {epoch + 1} of {epochs}')
            print("="*50)

            running_loss = 0.0
            self._model.train()  # training step initiated

            if self.verbose:
                print('Starting iteration over training dataset')

            iterable = tqdm(dataset, position=0,
                            leave=True) if self.verbose else dataset

            for batch_idx, (element) in enumerate(iterable):
                data = element[0].permute(0, 3, 1, 2).to(
                    device=DEVICE, dtype=torch.float32)
                targets = element[1].unsqueeze(1).to(
                    device=DEVICE, dtype=torch.float32)

                # forward
                with torch.cuda.amp.autocast():
                    predictions = self._model(data)
                    loss = loss_fn(predictions, targets)
                    dice = dice_score(predictions, targets)
                    print("AVERAGE DICE: ", np.sum(dice)/targets.shape[0])
                    # TODO Fix the loss function and plot
                    train_loss.append(loss.item())
                    total_loss = sum(loss for loss in train_loss)
                    pass

                # backward - calculating and updating the gradients of the network
                optimizer.zero_grad()
                # Compute gradients for each parameter based on the current loss calculation
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                avg_train_loss = total_loss/predictions.shape[0]
                running_loss += loss.item()*predictions.shape[0]

                # TRAINING STEP BEGINS FOR POSE ESTIMATION
                if self.pose_estimation:
                    keypoints = element[2]

                    if self.verbose:
                        print("="*50)
                        print("Generating training data for keypoint localization")
                        print("")
                    # generate pose data (VectorField)
                    poseData = crop_pose_data(data, targets)
                    vectorfield = VectorField(targets, data, keypoints)
                    trainPoseData = vectorfield.calculate_vector_field(
                        poseData[1], poseData[0], keypoints, poseData[2])

                    if self.verbose and epoch % 20 == 0:
                        vectorfield.visualize_gt_vectorfield(
                            trainPoseData[0], trainPoseData[1], imgIndx=-1)
                    
                    if self.save_images and epoch % 20 == 0:
                        img_meta = f"Epoch_{epoch},b_indx_{batch_idx}"
                        vectorfield.visualize_gt_vectorfield(
                            trainPoseData[0], trainPoseData[1], imgIndx=-1, saveImages=True, img_meta=img_meta)
                    
                     # Train the pose network
                    posePrediction = self.poseNetwork.train(
                        poseData, trainPoseData[0], trainPoseData[1], epoch_number=epoch)

            # The average loss of the epoch for segmentation
            epoch_losses.append(running_loss/batch_idx+1)

            if self.verbose:
                print("")
                print(
                    f"Average Train Loss for {self.model_name} epoch {epoch +1}: {avg_train_loss}")

            if self.verbose and epoch % 10 == 0:
                self.show_prediction(data, predictions)
            if self.save_images and epoch % 10 ==0:
                img_meta = f"Epoch_{epoch}"
                self.show_prediction(data, predictions, save_images=True, img_meta=img_meta)
            # ------ VALIDATION LOOP BEGINS -------

            if val_dataset is not None:
                running_val_loss = 0.0
                val_losses = []

                self._model.eval()  # Set model into evaluation mode

                with torch.no_grad():
                    if self.verbose:
                        print("="*50)
                        print("Starting iteration over validation dataset")

                    iterable = tqdm(val_dataset, position=0,
                                    leave=True) if self.verbose else val_dataset

                    for batch_idx, (element) in enumerate(iterable):
                        data = element[0].permute(0, 3, 1, 2).to(
                            device=DEVICE, dtype=torch.float32)
                        targets = element[1].unsqueeze(1).to(
                            device=DEVICE, dtype=torch.float32)

                        predictions = self._model(data)
                        dice = dice_score(predictions, targets)
                        val_loss = loss_fn(predictions, targets)
                        running_val_loss += val_loss.item() * \
                            predictions.shape[0]

                        if self.pose_estimation:
                            # generate pose data (VectorField)
                            keypoints = element[2]

                            if self.verbose:
                                print("="*50)
                                print(
                                    "Generating validation data for keypoint localization")
                                print("")
                            # generate pose data (VectorField)
                            poseData = crop_pose_data(data, targets)
                            vectorfield = VectorField(targets, data, keypoints)
                            trainPoseData = vectorfield.calculate_vector_field(
                                poseData[1], poseData[0], keypoints, poseData[2])

                            # set network to validation mode
                            posePrediction = self.poseNetwork.train(
                                poseData, trainPoseData[0], trainPoseData[1],epoch_number=epoch, phase=False)

                            # If epoch is 10, print a prediction
                    val_losses.append(running_val_loss/batch_idx+1)

                if epoch % 10 == 0:
                    # plot the losses
                    pass
        return losses

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

    def show_prediction(self, image, prediction, save_images=False, img_meta=""):
        # Show prediction
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        image = image[-1]
        # print(image.shape)
        prediction = prediction.detach().float().cpu().permute(0, 2, 3, 1).numpy()
        prediction = prediction[-1].squeeze()
        # print(prediction.shape)
        fig = plt.figure(figsize=(10, 10))
        img = fig.add_subplot(2, 3, 1)
        img.set_title("Image")
        img.imshow(image/255)

        pred = fig.add_subplot(2, 3, 2)
        pred.set_title("Prediction")
        pred.imshow(prediction)
        if save_images:
            plt.savefig(f"../results/{img_meta}")
        else:
            plt.show()

    def save(self, file):
        torch.save(self._model.state_dict(), file)
        if self.verbose:
            print("Model has been saved!")


# ----------------------------------------------------------------------------------------------
#                                     POSE ESTIMATION MODEL
# ----------------------------------------------------------------------------------------------


class PoseModel:

    DEFAULT = 'DCVnet'
    SGD = 'SGD'
    SCALER = torch.cuda.amp.GradScaler()

    def __init__(self, model=DEFAULT, device=None, keypoints=None, name=None, save_images=False, verbose=True):
        # Initialize the Pose Estimation Model
        # Assuming that the data is already loaded
        self.model = model
        self.name = name
        self.verbose = verbose
        self.keypoints = keypoints
        self.device = device
        self.save_images = save_images
        if self.device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        if self.model == self.DEFAULT:
            self.model = DCVnet()

    def train(self, images, vectorfield, epoch_number, keypoints=None, phase=True, learning_rate=0.005, optimizer=SGD, loss_fn=None, momentum=0.9, weight_decay=0.0005, gamma=0.1, lr_step_size=3, scaler=SCALER):
        # Takes in a list of tensors, the size of the batch_size set in DataLoader.
        DEVICE = self.device
        assert images is not None, "No image data has been received!"
        assert vectorfield is not None, "No Vectorfield data has been received"
        # Image is a list of tensors that has croped images of the container
        # vectorfield is a tensor containing the gt vectorfields found
        losses = []
        # Optimzer
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if optimizer in ["adam", "Adam"]:
            print("Pose Model Optimizer: Adam")
            optimizer = torch.optim.Adam(
                parameters, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer in ["sgd", "SGD"]:
            print("Pose Model Optimizer: SGD")
            optimizer = torch.optim.SGD(
                parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(
                f"The optimizer chosen: {optimizer}, is either not added yet or invalid.. please use SGD or Adam")

        # Send the model to the current device
        self.model.to(device=DEVICE)
        loss_fn = torch.nn.CrossEntropyLoss()
        #loss_fn = self.unit_loss_function()

        # ---------- STARTING TRAINING LOOP ------------
        if phase:
            if self.verbose:
                print("-"*50)
                print("Starting the training of DCVnet")
                print("-"*50)
                print("")

            for index, image in tqdm(enumerate(images[0])):
                assert torch.is_tensor(
                    image), f"The image is not a torch tensor!"
                self.model.train()  # Set model to training mode

                # Convert one by one the vectorfield gt to Tensor and rearrange so that the channels come first, send to the right device
                gtVf = torch.tensor(vectorfield[index]).permute(
                    2, 0, 1).to(device=DEVICE)

                with torch.cuda.amp.autocast():
                    predictions = self.model(image)
                    loss = self.huberloss_fn(predictions, gtVf)
                    losses.append(loss.item())
                    print("LOSSES: ", loss.item())

            # backward - calculating and updating the gradients of the network
            optimizer.zero_grad()
            # Compute gradients for each parameter based on the current loss calculation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if self.verbose:
                # Print the last vectorfield prediction with keypoints
                visualize_vectorfield(predictions, keypoints[index])
            if self.save_images:
                img_meta = f"Epoch_{epoch_number},b_indx_{index}"
                visualize_vectorfield(predictions, keypoints[index], saveImages=True, img_meta=img_meta)
        else:
            # ---------- STARTING VALIDATION LOOP ------------
            val_loss = 0.0
            self.model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                if self.verbose:
                    print("="*50)
                    print("Starting iteration over validation keypoint dataset")
                    print("")

                for index, image in tqdm(enumerate(images[0])):
                    assert torch.is_tensor(
                        image), f"The image is not a torch tensor!"

                    # Convert one by one the vectorfield gt to Tensor and rearrange so that the channels come first, send to the right device
                    gtVf = torch.tensor(vectorfield[index]).permute(
                        2, 0, 1).to(device=DEVICE)
                    # TODO Might need fixing

                    predictions = self.model(image)
                    loss = self.huberloss_fn(predictions, gtVf)
                    losses.append(loss.item())
                visualize_vectorfield(predictions, keypoints[index])
                if self.save_images:
                    img_meta = f"Epoch_{epoch_number},batch_indx_{index}"
                    visualize_vectorfield(predictions, keypoints[index], saveImages=True, img_meta=img_meta)


        return losses

    def huberloss_fn(self, prediction, target, delta=0.5):
        """
        Calculates the Huber Loss between the inputs

        args:
            prediction: The predicted vector field
            target: The ground truth vector field
            delta: the huber delta (Default: 0.5)

        return:
                loss: a scalar
        """

        # Check if this is more stable
        loss_fn = torch.nn.HuberLoss(delta=0.5)
        loss = loss_fn(prediction, target)
        #print("TORCH-HUBER: ", loss.item())
        #huberDelta = delta
        #loss = torch.abs(target-prediction)
        # loss = torch.where(loss < huberDelta, 0.5 * loss ** 2,
        #                   huberDelta * (loss - 0.5 * huberDelta))
        return loss

    def evaluate(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError


# Method to visualize keypoint prediction and rotation of container, Not implemented ye
    def show_prediction(self):
        raise NotImplementedError



    

    
