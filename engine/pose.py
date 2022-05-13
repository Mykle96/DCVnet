
# Packages
import os
import torch
import numpy as np
import torchvision
import random
from torch.utils.data import DataLoader
import torch.nn as nn
import warnings
from tqdm import tqdm


# local impors
from utils import utils
from vectorField import VectorField
from networks.vectorNetwork import DCVnet

# OLD FILE


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

    def train(self, images, vectorfield, keypoints, epoch_number, phase=True, learning_rate=0.005, optimizer=SGD, loss_fn=None, momentum=0.9, weight_decay=0.0005, gamma=0.1, lr_step_size=3, scaler=SCALER, name=None):
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
        # loss_fn = self.unit_loss_function()

        # ---------- STARTING TRAINING LOOP ------------
        if phase:

            for index, image in tqdm(enumerate(images[0])):
                assert torch.is_tensor(
                    image), f"The image is not a torch tensor!"
                self.model.train()  # Set model to training mode

                # Convert one by one the vectorfield gt to Tensor and rearrange so that the channels come first, send to the right device
                gtVf = torch.tensor(vectorfield[index]).permute(
                    2, 0, 1).to(device=DEVICE, dtype=torch.float32)

                with torch.cuda.amp.autocast():
                    predictions = self.model(image)
                    loss = self.huberloss_fn(predictions, gtVf)
                    losses.append(loss.item())

                # backward - calculating and updating the gradients of the network
                optimizer.zero_grad()
                # Compute gradients for each parameter based on the current loss calculation
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if self.verbose:
                # Print the last vectorfield prediction with keypoints
                #visualize_vectorfield(predictions, keypoints[index])
                pass
            if self.save_images and epoch_number % 10 == 0:
                img_meta = f"Epoch_{epoch_number},b_indx_{index}"
                visualize_vectorfield(
                    predictions, keypoints[index], saveImages=True, img_meta=img_meta)
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
                        2, 0, 1).to(device=DEVICE, dtype=torch.float32)
                    # TODO Might need fixing

                    predictions = self.model(image)
                    loss = self.huberloss_fn(predictions, gtVf)
                    losses.append(loss.item())
                visualize_vectorfield(predictions, keypoints[index])
                if self.save_images:
                    img_meta = f"Epoch_{epoch_number},batch_indx_{index}"
                    visualize_vectorfield(
                        predictions, keypoints[index], saveImages=True, img_meta=img_meta)

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
            filepath = "../models/DVFnet"

        torch.save(self.model.state_dict(), filepath+"/"+name)
        if self.verbose:
            print("*"*50)
            print("Pose Model has been saved!")
            print("*"*50)

# Method to visualize keypoint prediction and rotation of container, Not implemented ye

    def show_prediction(self):
        raise NotImplementedError
