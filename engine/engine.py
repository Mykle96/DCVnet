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

    def __init__(self, model=DEFAULT, classes=None, segmentation=True, pose_estimation=False, pretrained=False, verbose=True):
        # initialize the model class
        # If verbose is selected give more feedback of the process
        self.model_name = model
        self.pose_estimation = pose_estimation
        self.verbose = verbose
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
        BATCH_SIZE = len(dataset)
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
                f"The optimizer chosen: {optimizer}, is either not added yet or invalid.. please use SGD or Adam")
        # Initialize the loss function
        if loss_fn is None:
            if self.numClasses > 1:
                loss_fn = torch.nn.CrossEntropyLoss()
            if self.numClasses == 1:
                loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            loss_fn = loss_fn

        # LOAD CHECKPOINT

        # Set model to the correct device
        self._model.to(device=DEVICE)
        losses = []

        # TODO make a check on segmentation and if True, make another traning loop for BB
        # ----- TRAINING LOOP BEGINS -----
        print(f"Beginning traning with {self.model_name} network.")
        for epoch in tqdm(range(epochs)):

            print(f'Epoch {epoch + 1} of {epochs}')
            print("="*50)

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

                self.show_prediction(data, targets)
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

                    if self.verbose:
                        vectorfield.visualize_gt_vectorfield(
                            trainPoseData[0], trainPoseData[1])

                # forward
                with torch.cuda.amp.autocast():
                    predictions = self._model(data)
                    loss = loss_fn(predictions, targets)
                    if self.pose_estimation:
                        # Train the pose network
                        posePred = self.poseNetwork.train(
                            poseData, trainPoseData[0], trainPoseData[1])

                    # TODO Fix the loss function and plot
                    train_loss.append(loss.item())
                    total_loss = sum(loss for loss in train_loss)
                    avg_train_loss = total_loss/BATCH_SIZE

                # backward - calculating and updating the gradients of the network
                optimizer.zero_grad()
                # Compute gradients for each parameter based on the current loss calculation
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if self.verbose:
                print("")
                print(f"Average Train Loss: {avg_train_loss}")

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

    def __init__(self, model=DEFAULT, device=None, keypoints=None, name=None, verbose=True):
        # Initialize the Pose Estimation Model
        # Assuming that the data is already loaded
        self.model = model
        self.name = name
        self.verbose = verbose
        self.keypoints = keypoints
        self.device = device
        if self.device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        if self.model == self.DEFAULT:
            self.model = DCVnet()

    def train(self, images, vectorfield, keypoints=None, val_dataset=None, learning_rate=0.005, optimizer=SGD, loss_fn=None, momentum=0.9, weight_decay=0.0005, gamma=0.1, lr_step_size=3, scaler=SCALER):
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
        if self.verbose:
            print("-"*50)
            print("Starting the training of DCVnet")
            print("-"*50)
            print("")

        for index, image in tqdm(enumerate(images[0])):
            assert torch.is_tensor(image), f"The image is not a torch tensor!"
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

        print("Total loss: ", losses)
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
        loss_fn = torch.nn.HuberLoss(prediction, target)
        print("TORCH HUBER: ", loss_fn)
        huberDelta = delta
        loss = torch.abs(target-prediction)
        loss = torch.where(loss < huberDelta, 0.5 * loss ** 2,
                           huberDelta * (loss - 0.5 * huberDelta))
        return torch.sum(loss)

    def evaluate(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    containerLocalCords = np.array([

        (0, 0, 0),  # fbl
        (0, 0, -3),  # fbr
        (0, 3, 0),  # ftl
        (0, 3, -3),  # ftr
        (9, 0, 0),  # bbl
        (9, 0, -3),  # bbr
        (9, 3, 0),  # btl
        (9, 3, -3),  # btr
        (4.5, 1.5, -1.5)  # Center
    ])

    # Not sure about this array
    cameraIntrinsics = np.array(
        [[572.4114, 0., 325.2611], [0., 573.57043, 242.04899], [0., 0., 1.]])

    # Method to visualize keypoint prediction and rotation of container, Not implemented ye
    def show_prediction(self, vectorField, mask):

        success, R, t, predictions = self.predictPose(vectorField, mask)

        return R, t, predictions

        # Kjører pnp osv for å visualisere resultatet, tar inn ett bilde

    def predictPose(self, vectorField, mask, maskThreshold=0.9, localCords=containerLocalCords):

        if not(type(vectorField) == type(mask) == np.ndarray):
            vectorField = vectorField.toNumpy()
            mask = mask.toNumpy()

        maskCoordinates = np.where(mask > maskThreshold)[1:3]
        if not(len(maskCoordinates)):
            print("No coordinates in mask with probability value larger than threshold")
            return False, None

        hypDict = self.ransacVoting(maskCoordinates, vectorField)
        meanDict = self.getMean(hypDict)
        predictions = self.dictToArray(meanDict)

        return True, self.pnp(predictions), predictions

    def pnp(self, predictions, localCords=containerLocalCords, matrix=cameraIntrinsics, method=cv2.SOLVEPNP_ITERATIVE):

        try:
            _, R_exp, tVec = cv2.solvePnP(localCords,
                                          predictions,
                                          matrix,
                                          np.zeros(
                                              shape=[8, 1], dtype='float64'),
                                          flags=method)
        except Exception as e:
            print(e)
            # set_trace()
            print(predictions)
        return R_exp, tVec

    def ransacVoting(self, maskCoordinates, vectorField, numKeypoints=9, numHypotheses=5, ransacThreshold=.99):
        hypDict = {}
        for i in range(numKeypoints):
            hypDict[i] = []
        for n in range(numHypotheses):
            p1 = maskCoordinates.pop(random.randrange(len(maskCoordinates)))
            v1 = vectorField[p1[0]][p1[1]]
            p2 = maskCoordinates.pop(random.randrange(len(maskCoordinates)))
            v2 = vectorField[p2[0]][p2[1]]

            for i in range(numKeypoints):
                m1 = v1[i * 2 + 1] / v1[i * 2]  # get slopes
                m2 = v2[i * 2 + 1] / v2[i * 2]
                if not (m1 - m2):  # lines must intersect
                    print('slope cancel')
                    continue
                b1 = p1[0] - p1[1] * m1  # get y intercepts
                b2 = p2[0] - p2[1] * m2
                x = (b2 - b1) / (m1 - m2)
                y = m1 * x + b1
                weight = 0
                for voter in maskCoordinates:
                    yDiff = y - voter[0]
                    xDiff = x - voter[1]
                    mag = m.sqrt(yDiff ** 2 + xDiff ** 2)
                    vec = vectorField[voter[0]][voter[1]][i * 2: i * 2 + 2]

                    if self.ransacVal(yDiff / mag, xDiff / mag, vec) > ransacThreshold:
                        weight += 1
                hypDict[i].append(((y, x), weight))

                maskCoordinates.append(p1)
                maskCoordinates.append(p2)
        return hypDict

    def ransacVal(y1, x1, v2):  # dot product of unit vectors to find cos(theta difference)
        v2 = v2 / np.linalg.norm(v2)
        return y1 * v2[1] + x1 * v2[0]

    def getMean(hypDict):  # get weighted average of coordinates, weights list
        meanDict = {}
        for key, hyps in hypDict.items():
            xMean = 0
            yMean = 0
            totalWeight = 0
            for hyp in hyps:
                yMean += hyp[0][0] * hyp[1]
                xMean += hyp[0][1] * hyp[1]
                totalWeight += hyp[1]
            yMean /= totalWeight
            xMean /= totalWeight
            meanDict[key] = [yMean, xMean]
        return meanDict

    def dictToArray(hypDict):
        coordArray = np.zeros((len(hypDict.keys()), 2))
        for key, hyps in hypDict.items():
            coordArray[key] = np.array(
                [round(hyps[1]), round(hyps[0])])  # x, y format
        return coordArray
