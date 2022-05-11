from turtle import Vec2D
import torch
from torch import Tensor
from cv2 import cv2
import numpy as np
import math as m
import colorsys
import sys
import random
import os
import torchvision
# from dataLoader import ShippingDataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TVF
import matplotlib.pyplot as plt
# from DCVnet.engine.engine import Model


"""
Utility file containing a lot of funct ions used across the system
"""

#----------------------#
# MATHEMATICAL UTILS
#----------------------#


#----------------------#
# PERFORMANCE UTILS
#----------------------#


def dice_score(prediction: Tensor, target: Tensor, classes=None, threshold=0.5, epsilon=1e-6, **kwargs):
    """
    Function for calculating the overlapping area of intersection. The closer to one the greater the overlap and
    subsequently the better the accuracy.

    args:
        prediction - the prediction masks given from the model - Tensor [BATCH, channels, H, W]
        target - the ground truth mask - Tensor [BATCH, channels, H, W]
        classes - the number of classes in the image
        threshold - The minimum accepted threshold of overlap (default at 0.5)

    return:
        Returns a list of float32s between the threshold and 1 (i.e default is values between [0.5, 1])
    """

    assert prediction.size() == target.size(
    ), "The prediction and target input do not have matching sizes! "
    dice = []
    # covnert predictions to probabilities using sigmoid
    prediction = torch.sigmoid(prediction)

    for i in range(prediction.shape[0]):  # Exluding the background (0)
        # Remove batch and channel dimentions, BATCH x 1 x H X W =>
        pred = prediction[i].squeeze(1
                                     ).numpy().astype(int) if prediction[i].dim() > 3 else prediction[i].numpy().astype(int)

        mask = target[i].squeeze(1).numpy(
        ).astype(int) if target[i].dim() > 3 else target[i].numpy().astype(int)
        # visualize_croped_data(pred, mask)
        intersection = (pred & mask).sum((1, 2))

        union = (pred | mask).sum((1, 2))

        diceScore = float((intersection+epsilon))/float((union + epsilon))

        if diceScore < threshold:
            print("Iou score was below the predetermined threshold of {}, and was thus purged from the set".format(
                threshold))
            dice.append(float('nan'))

        else:
            print(f"Dice Score for prediction {i}: {diceScore}")
            dice.append(diceScore)
    return np.array(dice)


def plot_loss(train_loss=None, val_loss=None, epochs=None):
    # takes in two lists and the number of epochs that have runned
    assert len(train_loss) and len(val_loss) is not 0
    fig = plt.figure()
    loss = fig.add_subplot(121, title="losses")
    loss.plot(epochs, train_loss, 'bo-', label='train loss')
    loss.plot(epochs, val_loss, 'ro-', label='validation loss')
    plt.plot(loss)


def validation_loss():
    raise NotImplementedError("Not yet implemented")


def training_loss():
    raise NotImplementedError("Not yet implemented")


def average_loss():
    raise NotImplementedError("Not yet implemented")


def total_loss():
    raise NotImplementedError("Not yet implemented")


def hot_encoder():
    raise NotImplementedError("Not yet implemented")


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


#----------------------#
# PIPELINE UTILS
#----------------------#

def crop_from_prediction(image, prediction, threshold=0.6):
    """
    Function for croping an images on the predicted masks. Crops the image at a threshold to
    eliminate outliers in the prediction.

    args:
        image: Tensor of images
        mask: Tensor of masks
        threshold: threshold for filtering weak predicitons (Default: 0.6)

    return:
        cropedImage: a croped image by using the predicted mask

    """
    pass


#----------------------#
# NETWORK UTILS
#----------------------#


def visualize_vectorfield(field, keypoint, indx=5, oneImage=True):
    '''
    Function to visualize vector field towards a certain keypoint, and plotting all keypoint in subplot

    args:
        field:  Tensor with vector field for an image, shape [1, 2*num keypoints, dimX, dimY]
        keypoint: Tensor with the format (number of keypoints, 2)
        indx: keypoint index, default is last keypoint 

    returns:
        No return
    '''

    if not keypoint == None:

        # Transforms field to numpy array and to the shape (dimY, dimX, 2*num_keypoints)
        if not isinstance(field, np.ndarray):
            field = field.permute(0, 2, 3, 1).detach().squeeze().numpy()
        keypoint = keypoint.numpy()

        dimentions = [field.shape[0], field.shape[1]]  # y,x
        numCords = int(field.shape[2]/2)

        if not(oneImage):
            rows = m.ceil(m.sqrt(numCords))
            cols = m.ceil(m.sqrt(numCords))
            fig, ax = plt.subplots(
                rows, cols, figsize=(10, 10))  # ax[y,x] i plottet
            ax = ax.flatten()
            numImages = numCords
        else:
            numImages = 1
            if(indx == -1):
                indx = numCords-1
            elif(indx > numCords or indx < 0):
                raise ValueError(
                    f"Keypoint value = {indx} needs to be in the interval [1, number of keypoints = {numCords}]")

        for index in range(numImages):
            newImg = np.zeros((dimentions[0], dimentions[1], 3))
            if(oneImage):
                index = indx

            for i in range(dimentions[0]):
                for j in range(dimentions[1]):
                    if(field[i][j] != np.zeros(2*numCords)).all():

                        cy = j
                        cx = i
                        x = cx + 2*field[i][j][2*index]
                        y = cy + 2*field[i][j][2*index+1]

                        if(cx-x) < 0:
                            # (2) og (3)
                            angle = m.atan((cy-y)/(cx-x))+m.pi
                        elif(cy-y) < 0:
                            # (4)
                            if(cx == x):
                                # 270 grader
                                angle = 3/2*m.pi
                            else:
                                angle = m.atan((cy-y)/(cx-x))+2*m.pi
                        else:
                            # (1)
                            if(cx == x):
                                angle = m.pi/2
                            else:
                                angle = m.atan((cy-y)/(cx-x))

                        h = angle/(2*m.pi)
                        rgb = colorsys.hsv_to_rgb(h, 1.0, 1.0)
                        newImg[i][j] = rgb

            if not(oneImage):
                ax[index].imshow(newImg)

            for k in range(numCords):
                if k == index:
                    marker = '+'
                    color = 'white'
                else:
                    marker = '.'
                    color = 'black'

                if not(oneImage):
                    ax[index].plot(dimentions[1]*keypoint[k][0], dimentions[0]-keypoint[k][1] *
                                   dimentions[0], marker=marker, color=color)
                else:
                    plt.plot(dimentions[1]*keypoint[k][0], dimentions[0]-keypoint[k][1] *
                             dimentions[0], marker=marker, color=color)

        if(oneImage):
            plt.imshow(newImg)

        plt.show()
    else:
        pass


#----------------------#
# TRAINING UTILS
#----------------------#

def visualize_croped_data(crop_image, crop_mask):
    # function for visualizing the croped image and mask

    crop = torch.squeeze(crop_image) if crop_image.dim() > 3 else crop_image
    crop_mask = torch.squeeze(crop_mask) if crop_mask.dim() > 2 else crop_mask
    crop = crop*255
    crop_mask = crop_mask*255
    crop_mask_img = torchvision.transforms.ToPILImage()(crop_mask)
    img = torchvision.transforms.ToPILImage()(
        crop) if crop.dim() == 3 else torchvision.transforms.ToPILImage()(crop)
    img.show()
    crop_mask_img.show()


def crop_pose_data(image, mask, threshold=0.6):
    """
    Function for croping the training (and validation) images and masks. Crops the mask at a threshold to
    eliminate outliers.

    args:
        image: Tensor of images - [batch,channels,h,w]
        mask: Tensor of masks - [h,w]
        threshold: threshold for filtering weak predicitons (Default: 0.6)

    return:
        poseData: list of images, masks and respective coordinfo.

    """
    # initialize variables
    cropedImages = []
    cropedMask = []
    coordInfo = []

    for i in range(len(image)):
        # Fetch coordinates of mask wiht a threshold
        coords = np.where(mask[i] >= threshold)[1:3]
        # Setting top coordinate of the crop, the largest corner of the mask
        top_y = (min(coords[0]) - 10) if min(coords[0]) > 10 else 0
        top_x = min(coords[1]) - 10 if min(coords[1]) > 10 else 0
        # setting the width and height accrording to the biggest corner of the mask
        height = max(coords[0])-top_y + 20 if max(coords[0])-top_y < 580 else 0
        width = max(coords[1]) - top_x + \
            20 if max(coords[1]) - top_x < 580 else 0

        # make sure the dimentions are even numbers for the neural network
        if not height % 2 == 0:
            height += 1
        if not width % 2 == 0:
            width += 1

        # Add the new metrics to a index dependant list
        info = [top_x, top_y, height, width]
        coordInfo.append(info)

        # Crop the image and mask on the given point and dimentions
        crop, crop_mask = TVF.crop(
            image[i], top_y, top_x, height, width), TVF.crop(
            mask[i], top_y, top_x, height, width)

        # unsqueeze to add batch dimention and append the cropped images and masks
        cropedImages.append(torch.unsqueeze(
            crop, 0))
        cropedMask.append(torch.unsqueeze(crop_mask, 0))

    # Visualize the last image and mask
    visualize_croped_data(crop, crop_mask)
    # List of lists
    poseData = [cropedImages, cropedMask, coordInfo]

    return poseData


def save_checkpoint(state, filename, **kwargs):
    print("==> Saving checkpoint")
    # TODO
    if "print" in kwargs:
        raise NotImplementedError(
            "this print function has not been implemented yet")
    torch.save(state, filename)


def load_checkpoint(model, checkpoint):
    assert type(checkpoint) is dict, ValueError(
        f"Expected checkpoint to be a dictionary, got {type(checkpoint)} instead!")
    print("==> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dictionary"])


def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()


def test():
    imagePath = "../data/dummy-data/valid-mask/4_id copy.png"
    maskPath = "../data/dummy-data/valid-mask/4_id.png"
    image = plt.imread(imagePath)
    mask = plt.imread(maskPath)
    plt.imshow(image)
    # plt.show()
    plt.imshow(mask)
    # plt.show()
    intersection_over_union(image, mask, 1)


if __name__ == "__main__":
    test()
