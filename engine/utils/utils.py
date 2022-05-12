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
    prediction = prediction if classes > 1 else torch.sigmoid(prediction)

    for i in range(prediction.shape[0]):  # Exluding the background (0)
        # Remove batch and channel dimentions, BATCH x 1 x H X W =>
        pred = prediction[i].detach().flatten().cpu().numpy().astype(
            int) if prediction[i].dim() > 3 else prediction[i].detach().cpu().numpy().astype(int)

        mask = target[i].detach().flatten().cpu().numpy(
        ).astype(int) if target[i].dim() > 3 else target[i].detach().cpu().numpy().astype(int)
        # visualize_croped_data(pred, mask)
        intersection = np.sum(mask * pred)
        union = np.sum(mask) + np.sum(pred)

        diceScore = (2*intersection+epsilon)/(union + epsilon)

        if diceScore < threshold:
            # print("Iou score was below the predetermined threshold of {}, and was thus purged from the set".format(
            #    threshold))
            dice.append(0.0)

        else:
            # print(f"Dice Score for prediction {i}: {diceScore}")
            dice.append(diceScore)
    return np.array(dice)


def plot_loss(train_loss=None, val_loss=None, epochs=None, name=None):
    # takes in two lists and the number of epochs that have runned
    assert len(train_loss) and len(val_loss) is not 0
    fig = plt.figure(figsize=(10, 5))
    plt.title(f"{name} Training and Validation Loss")
    plt.plot(train_loss, 'bo-', label='train loss')
    plt.plot(val_loss, 'ro-', label='validation loss')
    plt.legend(['Train', 'Valid'])
    plt.show()


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


# Define constants used in PNP/Ransac methods
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

# Not sure about this array TODO
cameraIntrinsics = np.array(
    [[572.4114, 0., 325.2611], [0., 573.57043, 242.04899], [0., 0., 1.]])


def predictPose(vectorField, mask=None, maskThreshold=0.9, localCords=containerLocalCords):

    if not mask == None:  # Håndter hvis ikke mask

        if not(type(vectorField) == type(mask) == np.ndarray):
            vectorField = vectorField.toNumpy()
            mask = mask.toNumpy()

        maskCoordinates = np.where(mask > maskThreshold)[1: 3]
        if not(len(maskCoordinates)):
            print("No coordinates in mask with probability value larger than threshold")
            return False, None
    else:
        maskCoordinates = (
            np.arange(0, vectorField.shape[2], dtype='int'),
            np.arange(0, vectorField.shape[3], dtype='int'))

    hypDict = ransacVoting(maskCoordinates, vectorField)
    meanDict = getMean(hypDict)
    predictions = dictToArray(meanDict)

    return True, pnp(predictions), predictions


def pnp(predictions, localCords=containerLocalCords, matrix=cameraIntrinsics, method=cv2.SOLVEPNP_ITERATIVE):

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


def ransacVoting(maskCoordinates, vectorField, numKeypoints=9, numHypotheses=5, ransacThreshold=.99):
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

                if ransacVal(yDiff / mag, xDiff / mag, vec) > ransacThreshold:
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

#----------------------#
# NETWORK UTILS
#----------------------#


def visualize_vectorfield(field, keypoint, indx=-1, oneImage=True):
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
            field = field.permute(
                0, 2, 3, 1).detach().squeeze().cpu().numpy()
        keypoint = keypoint.cpu().numpy()

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
        coords = torch.where(mask[i] >= threshold)[1:3]
        print("COORDS: ", coords)
        # Setting top coordinate of the crop, the largest corner of the mask
        top_y = (min(coords[0]) - 10) if min(coords[0]
                                             ) > 10 else 0
        top_x = min(coords[1]) - 10 if min(coords[1]) > 10 else 0
        # setting the width and height accrording to the biggest corner of the mask
        height = max(coords[0])-top_y + 20 if max(coords[0]
                                                  )-top_y < 580 else max(coords[0])-top_y
        width = max(coords[1]) - top_x + \
            20 if max(coords[1]) - top_x < 580 else max(coords[1]) - top_x

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


if __name__ == "__main__":
    test()
