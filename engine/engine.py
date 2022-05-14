# packages
from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn
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
from networks.network import UNET
from networks.vectorNetwork import DCVnet
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
    default_classes = ['container']

    def __init__(self, model=DEFAULT, classes=default_classes, device=None, save_images=False, verbose=True):
        # initialize the model class
        # If verbose is selected give more feedback of the process
        self.model_name = model
        self._device = device
        self.verbose = verbose
        self.save_images = save_images
        self.classes = classes
        self.numClasses = len(classes)
        self.multiple_gpu = bool()
        self.num_gpu = int()
        self.device_ids = []

        # Check for devices and set them
        if self._device is None:
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
                self.multiple_gpu, self.num_gpu = gpu_check()
                for i in range(self.num_gpu):
                    self.device_ids.append(i)

                torch.cuda.empty_cache()
            else:
                self._device = torch.device("cpu")
        else:
            self._device = device

        if model == self.DEFAULT:
            self._model = UNET()

    def train(self, dataset, val_dataset=None, epochs=150, learning_rate=0.005, optimizer=ADAM, loss_fn=None, momentum=0.9, weight_decay=0.0005, gamma=0.1, lr_step_size=3, scaler=SCALER, name=None):
        DEVICE = self._device

        if DEVICE == torch.device('cpu'):
            print("")
            warnings.warn('It looks like you\'re training your model on a CPU. '
                          'Consider switching to a GPU as this is not recommended.', RuntimeWarning)
            print("")
        else:
            # Check for multiple CUDA, if true, set to parallell processing
            if self.multiple_gpu:
                # Sending model to multiple GPUs
                if self.verbose:
                    print(
                        "Multiple GPUs registered, initiating Distributed Data Parallel")
                    #self.model = nn.parallel.DistributedDataParallel(self.model)
                self._model = nn.DataParallel(
                    self._model, device_ids=self.device_ids)
                DEVICE = f"{DEVICE}"+':'+f"{self._model.device_ids[0]}"
                self._model.to(device=DEVICE)
            else:
                # If not --> set model to device
                self._model.to(device=DEVICE)

        # Check if the dataset is converted or not, if not initiate, also check for any validation sets.
        assert dataset is not None, "No dataset was received, make sure to input a dataset"
        if not isinstance(dataset, DataLoader):
            dataset = DataLoader(dataset, shuffle=True)

        if val_dataset is not None and not isinstance(val_dataset, DataLoader):
            val_dataset = DataLoader(val_dataset, shuffle=True)

        # Select optimizer and tune parameters
        assert type(
            optimizer) == str, f"Error catched for the optimizer parameter! Expected the input to be of type string, but got {type(optimizer)}."
        # Get parameters that have grad turned on (i.e. parameters that should be trained)
        parameters = [p for p in self._model.parameters() if p.requires_grad]

        if optimizer in ["adam", "Adam", "ADAM"]:
            print("Optimizer: Adam")
            optimizer = torch.optim.Adam(
                parameters, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer in ["sgd", "SGD", "Sgd"]:
            print("Optimizer: SGD")
            optimizer = torch.optim.SGD(
                parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(
                f"The optimizer chosen: {optimizer}, is either not added yet or invalid.. please use SGD or Adam")
        # Adding a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_step_size, gamma=gamma)

        # Initialize the loss function
        if loss_fn is None:
            if self.numClasses > 1:
                loss_fn = torch.nn.CrossEntropyLoss()
                print("Loss Function: ", loss_fn)

            if self.numClasses == 1:
                loss_fn = torch.nn.BCEWithLogitsLoss()
                print("Loss Function: ", loss_fn)
        else:
            loss_fn = loss_fn

        # LOAD CHECKPOINT
        train_loss = []
        epoch_losses = []
        losses = []
        val_losses = []

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
                data = element[0].to(
                    device=DEVICE, dtype=torch.float32)
                targets = element[1].to(
                    device=DEVICE, dtype=torch.float32)

                # forward
                with torch.cuda.amp.autocast():
                    predictions = self._model(data)
                    # If Cross entropy is used, add Softmax on the pred before loss is calculated, Update sigmoid to softmax
                    pred = torch.sigmoid(
                        predictions) if loss_fn == torch.nn.CrossEntropyLoss() else predictions
                    loss = loss_fn(pred, targets)
                    # dice = dice_score(pred, targets, self.numClasses)

                # backward - calculating and updating the gradients of the network
                optimizer.zero_grad()
                # Compute gradients for each parameter based on the current loss calculation
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # avg_train_loss = train_loss/predictions.shape[0]
                running_loss += loss.item()*predictions.shape[0]

            if self.save_images and (epoch+1) % 50 == 0:
                img_meta = f"Epoch_{epoch},b_indx_{batch_idx}"
                vectorfield.visualize_gt_vectorfield(
                    trainPoseData[0], trainPoseData[1], imgIndx=-1, saveImages=True, img_meta=img_meta)

            # The average loss of the epoch for segmentation
            epoch_losses.append(running_loss/batch_idx+1)
            train_loss.append(running_loss/batch_idx+1)

            if self.verbose:
                print("")
                print(
                    f"Average Train Loss for {self.model_name} epoch {epoch +1}: {running_loss/batch_idx+1}")
                print("")

            if self.verbose and (epoch+1) % 10 == 0:
                if loss_fn == torch.nn.CrossEntropyLoss():
                    if(self.save_images):
                        img_meta = f"Epoch_{epoch}"
                        self.show_prediction(
                            data, pred, save_images=True, img_meta=img_meta)
                    else:
                        self.show_prediction(data, pred)
                else:
                    if(self.save_images):
                        img_meta = f"unet_Training_pred_Epoch_{epoch}"
                        self.show_prediction(data, torch.sigmoid(
                            pred), save_images=True, img_meta=img_meta)
                    else:
                        self.show_prediction(data, torch.sigmoid(pred))

            # ------ VALIDATION LOOP BEGINS -------

            if val_dataset is not None:
                running_val_loss = 0.0

                self._model.eval()  # Set model into evaluation mode

                with torch.no_grad():
                    if self.verbose:
                        print("="*50)
                        print("Starting iteration over validation dataset")

                    iterable = tqdm(val_dataset, position=0,
                                    leave=True) if self.verbose else val_dataset

                    for batch_idx, (element) in enumerate(iterable):
                        data = element[0].to(
                            device=DEVICE, dtype=torch.float32)
                        targets = element[1].to(
                            device=DEVICE, dtype=torch.float32)

                        predictions = self._model(data)
                        # Update sigmoid to softmax
                        pred = torch.sigmoid(
                            predictions) if loss_fn == torch.nn.CrossEntropyLoss() else predictions
                        val_dice = dice_score(pred, targets, 2)
                        val_loss = loss_fn(pred, targets)

                        running_val_loss += val_loss.item() * \
                            predictions.shape[0]

                val_losses.append(running_val_loss/batch_idx+1)
                print("")
                print("AVERAGE VAL DICE: ",
                      np.sum(val_dice)/targets.shape[0])
                print(
                    f"Average Validation Loss for {self.model_name} epoch {epoch +1}: {running_val_loss/batch_idx+1}")
                print("")

                if (epoch+1) % 10 == 0:
                    if loss_fn == torch.nn.CrossEntropyLoss():
                        self.show_prediction(data, pred)
                    else:
                        self.show_prediction(data, torch.sigmoid(pred))
                    plot_loss(epoch_losses, val_losses, epoch+1)
                    # Update the learning rate every few epochs
            lr_scheduler.step()
        # Save the model
        if name is None:
            name = self.model_name + f"_v.{random.randint(0,10)}"
        else:
            name = name
        self.save(name=name)
        return epoch_losses

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
            path = 'DCVnet/results'
            plt.savefig(path + "/" + img_meta+".png")
        else:
            plt.show()

    def save(self, filepath=None, name=None):
        if filepath is None or not os.path.exists(filepath):
            filepath = "DCVnet/models/unet"

        torch.save(self._model.state_dict(), filepath+"/"+name)
        if self.verbose:
            print("="*50)
            print("Model has been saved!")
            print("="*50)


# ----------------------------------------------------------------------------------------------
#                                     POSE ESTIMATION MODEL
# ----------------------------------------------------------------------------------------------


class VectorModel():
    # Default values
    DEFAULT = "DCVnet"
    SGD = 'sgd'
    SCALER = torch.cuda.amp.GradScaler()

    def __init__(self, model=DEFAULT, device=None, name=None, save_images=False, verbose=True):

        self.model = model
        self.name = name
        self.verbose = verbose
        self.device = device
        self.save_images = save_images
        self.multiple_gpu = bool()
        self.num_gpu = int()
        self.device_ids = []

        # Check for devices and set them
        if self.device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.multiple_gpu, self.num_gpu = gpu_check()
                for i in range(self.num_gpu):
                    self.device_ids.append(i)

                torch.cuda.empty_cache()
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        if self.name is None:
            self.name = self.model

        if self.model == self.DEFAULT:
            self.model = DCVnet()

    def train(self, dataset, val_dataset=None, epochs=None, learning_rate=0.005, optimizer=SGD, loss_fn=None, momentum=0.9, weight_decay=0.0005, gamma=0.1, lr_step_size=3, scaler=SCALER, name=None):

        DEVICE = self.device

        assert dataset is not None, "No dataset was received, make sure to input a dataset"
        if not isinstance(dataset, DataLoader):
            dataset = DataLoader(dataset, shuffle=True)

        if val_dataset is not None and not isinstance(val_dataset, DataLoader):
            val_dataset = DataLoader(val_dataset, shuffle=False)

        if self._device == torch.device('cpu'):
            print("")
            warnings.warn('It looks like you\'re training your model on a CPU. '
                          'Consider switching to a GPU as this is not recommended.', RuntimeWarning)
            print("")
        else:
            # Check for multiple CUDA, if true, set to parallell processing
            if self.multiple_gpu:
                # Sending model to multiple GPUs
                if self.verbose:
                    print(
                        "Multiple GPUs registered, initiating Distributed Data Parallel")
                    #self.model = nn.parallel.DistributedDataParallel(self.model)
                self.model = nn.DataParallel(
                    self.model, device_ids=self.device_ids)
                DEVICE = f"{DEVICE}"+':'+f"{self.model.device_ids[0]}"
                self.model.to(device=DEVICE)
            else:
                # If not --> set model to device
                self.model.to(device=DEVICE)

        # Optimzer
        parameters = [p for p in self.model.parameters() if p.requires_grad]

        if optimizer in ["adam", "Adam", "ADAM"]:
            print("Optimizer: Adam")
            optimizer = torch.optim.Adam(
                parameters, lr=learning_rate, weight_decay=weight_decay)

        elif optimizer in ["sgd", "SGD", "Sgd"]:
            print("Optimizer: SGD")
            optimizer = torch.optim.SGD(
                parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(
                f"The optimizer chosen: {optimizer}, is either not added yet or invalid.. please use SGD or Adam")

        # Add learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_step_size, gamma=gamma)
        # Check for loss functions
        if loss_fn is not None:
            loss_fn = loss_fn

        train_losses = []
        val_losses = []

        print(f" --- Starting traininig on model: {self.name} ---")
        # ---------- STARTING TRAINING LOOP ------------
        for epoch in tqdm(range(epochs)):

            print(f'Epoch {epoch + 1} of {epochs}')
            print("="*50)

            if self.verbose:
                print("Starting iteration over training dataset")
                print("and calculating unit vector fields")
                print("")

            iterable = tqdm(dataset, position=0,
                            leave=True) if self.verbose else dataset

            running_loss = 0.0

            for batch_idx, (element) in enumerate(iterable):
                assert len(
                    element) == 3, f"Data deprication! Expected {element} to have three elements, instead got {len(element)}"
                data = element[0].to(
                    device=DEVICE, dtype=torch.float32)
                targets = element[1].to(
                    device=DEVICE, dtype=torch.float32)
                keypoints = element[2]

                self.model.train()  # Set model to training mode
                # crop incoming images on their masks
                poseData = crop_pose_data(data, targets)
                vectorfield = VectorField(targets, data, keypoints)
                # calculate the ground truth vector fields
                trainPoseData = vectorfield.calculate_vector_field(
                    poseData[1], poseData[0], keypoints, poseData[2])

                gtVfList = trainPoseData[0]
                keypoints = trainPoseData[1]

                for index, image in enumerate(poseData[0]):
                    assert torch.is_tensor(
                        image), f"The image is not a torch tensor!"
                    # Convert one by one the vectorfield gt to Tensor and rearrange so that the channels come first, send to the right device
                    gtVf = torch.tensor(gtVfList[index]).permute(
                        2, 0, 1).to(device=DEVICE, dtype=torch.float32)

                    with torch.cuda.amp.autocast():
                        predictions = self.model.forward(image)
                        loss = self.huberloss_fn(
                            predictions, gtVf) if loss_fn is None else loss_fn

                    # backward - calculating and updating the gradients of the network
                    optimizer.zero_grad()
                    # Compute gradients for each parameter based on the current loss calculation
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    running_loss += loss.item()

                    if self.verbose and (batch_idx+1) % 50 == 0:
                        vectorfield.visualize_gt_vectorfield(
                            trainPoseData[0], trainPoseData[1], imgIndx=-1)

                    if self.save_images and (batch_idx+1) % 25 == 0:
                        img_meta = f"Epoch_{epoch},b_indx_{batch_idx}"
                        vectorfield.visualize_gt_vectorfield(
                            trainPoseData[0], trainPoseData[1], imgIndx=-1, saveImages=True, img_meta=img_meta)

            train_losses.append(running_loss/index+1)
            print("")
            print(
                f"Average Train Loss for {self.name} epoch {epoch +1}: {running_loss/(index+1)}")
            print("")
            if self.save_images and epoch % 10 == 0:
                img_meta = f"Epoch_{epoch},b_indx_{index}"
                visualize_vectorfield(
                    predictions, keypoints[index], saveImages=True, img_meta=img_meta)

                # ---------- STARTING VALIDATION LOOP ------------
            if val_dataset is not None:
                running_val_loss = 0.0
                self.model.eval()  # Set model to evaluation mode
                with torch.no_grad():
                    if self.verbose:
                        print("="*50)
                        print("Starting iteration over validation dataset")
                        print("and calculating unit vector fields")
                        print("")

                    iterable = tqdm(dataset, position=0,
                                    leave=True) if self.verbose else dataset
                    for batch_idx, (element) in enumerate(iterable):
                        data = element[0].to(
                            device=DEVICE, dtype=torch.float32)
                        targets = element[1].to(
                            device=DEVICE, dtype=torch.float32)

                        poseData = crop_pose_data(data, targets)
                        vectorfield = VectorField(targets, data, keypoints)
                        trainPoseData = vectorfield.calculate_vector_field(
                            poseData[1], poseData[0], keypoints, poseData[2])

                        gtVfList = trainPoseData[0]
                        keypoints = trainPoseData[1]

                        for index, image in enumerate(poseData[0]):
                            assert torch.is_tensor(
                                image), f"The image is not a torch tensor!"

                            # Convert one by one the vectorfield gt to Tensor and rearrange so that the channels come first, send to the right device
                            gtVf = torch.tensor(gtVfList[index]).permute(
                                2, 0, 1).to(device=DEVICE, dtype=torch.float32)

                            predictions = self.model(image)
                            loss = self.huberloss_fn(
                                predictions, gtVf) if loss_fn is None else loss_fn

                            running_val_loss += loss.item()

                    if self.save_images:
                        img_meta = f"Epoch_{epoch},batch_indx_{index}"
                        visualize_vectorfield(
                            predictions, keypoints[index], saveImages=True, img_meta=img_meta)
                val_losses.append(running_val_loss/index+1)
                print("")
                print(
                    f"Average Train Loss for {self.name} epoch {epoch +1}: {running_val_loss/(index+1)}")
                print("")
            # update lr
            lr_scheduler.step()
            if (epoch+1) % 10 == 0:
                # plot the loss and validation
                plot_loss(train_loss=train_losses,
                          val_loss=val_losses, epochs=epoch+1)
        # Save model
        if name is None:
            name = self.model_name + f"_v.{random.randint(0,10)}"
        else:
            name = name
        self.save(name=name)

        return val_losses

    def huberloss_fn(self, prediction=None, target=None, delta=0.5):
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
        loss_fn = torch.nn.HuberLoss(delta=0.5)  # Maybe lower the delta
        target, prediction = target.squeeze(), prediction.squeeze()  # Remove batch dimention
        loss = loss_fn(prediction, target)
        # print("TORCH-HUBER: ", loss.item())
        # huberDelta = delta
        # loss = torch.abs(target-prediction)
        # loss = torch.where(loss < huberDelta, 0.5 * loss ** 2,
        #                   huberDelta * (loss - 0.5 * huberDelta))
        return loss

    def evaluate(self):
        raise NotImplementedError

    def save(self, filepath=None, name=None):
        if name is None:
            name = "Pose_network" + f"_v.{random.randint(0,10)}"
        else:
            name = name
        if filepath is None or not os.path.exists(filepath):
            filepath = "DCVnet/models/DVFnet"

        torch.save(self.model.state_dict(), filepath+"/"+name)
        if self.verbose:
            print("*"*50)
            print("Pose Model has been saved!")
            print("*"*50)

    def show_prediction(self):
        raise NotImplementedError
