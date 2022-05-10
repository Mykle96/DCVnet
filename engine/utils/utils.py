import torch
from cv2 import cv2
import numpy as np
import math as m
import colorsys
import sys
import random
import os
import torchvision
#from dataLoader import ShippingDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
#from DCVnet.engine.engine import Model


"""
Utility file containing a lot of functions used across the system
"""

#----------------------#
# MATHEMATICAL UTILS
#----------------------#


#----------------------#
# PERFORMANCE UTILS
#----------------------#


def intersection_over_union(prediction, target, classes, threshold=0.5, **kwargs):
    """
    Function for calculating the overlapping area of intersection. The closer to one the greater the overlap and
    subsequently the better the accuracy.

    args:
        prediction - the prediction mask given from the model
        target - the ground truth mask
        classes - the number of classes in the image
        threshold - The minimum accepted threshold of overlap (default at 0.5)

    return:
        Returns a list of float32s between the threshold and 1 (i.e default is values between [0.5, 1])
    """
    ious = []
    prediction = prediction.view(-1)
    target = target.view(-1)

    for cls in range(1, classes):  # Exluding the background (0)
        predictionInds = prediction == cls
        targetInds = target == cls
        intersection = (predictionInds[targetInds]).long().sum().data.cpu()[0]
        union = predictionInds.long().sum().data.cpu(
        )[0] + targetInds.long().sum().data.cpu()[0] - intersection
        if union == 0:
            print("No ground truth was found ---> removing the test from the evaluation")
            ious.append(float('nan'))
        else:
            iou = float(intersection)/float(max(union, 1))
            if iou < threshold:
                print("Iou score was below the predetermined threshold of {}, and was thus purged from the set".format(
                    threshold))
            else:
                print(f"Dice Score: {iou}")
                ious.append(iou)

    return np.array(ious)


def validation_loss():
    raise NotImplementedError("Not yet implemented")


def training_loss():
    raise NotImplementedError("Not yet implemented")


def average_loss():
    raise NotImplementedError("Not yet implemented")


def total_loss():
    raise NotImplementedError("Not yet implemented")


def dice_score(prediction, target, **kwargs):
    """
    Function for calculating the dice score (Jaccard index)
    """

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
# DATASET UTILS
#----------------------#


#----------------------#
# NETWORK UTILS
#----------------------#


def visualize_vectorfield(field, keypoint, indx=-1):
    # Takes in a np_array of the found unit vector field and displays it
    # Keypoint is a single keypoint on the format (x,y)
    # Field is a vectorfield on the format (600,600,18)

    # trainPoseData [5,600,600,18] tensor
    # keypoint (5,9,2) tensor
    print("Utils fieldtype :", type(field))
    print("Utils visulize shape: ", keypoint)
    if not isinstance(field, np.ndarray):
        field = field.permute(0, 2, 3, 1).detach().squeeze().numpy()
    keypoint = keypoint.numpy()
    print("Utils fieldshape :", type(field))
    print("Utils fieldshape :", field.shape)

    #field = field[imgInt]
    #all_keypoints = keypoint
    #keypoint = keypoint[indx]
    dimentions = [field.shape[0], field.shape[1]]  # y,x
    print("DIMENTIONS: ", dimentions)

    newImg = np.zeros((dimentions[0], dimentions[1], 3))
    numCords = int(field.shape[2]/2)

    for i in range(dimentions[0]):
        for j in range(dimentions[1]):
            if(field[i][j] != np.zeros(2*numCords)).all():

                cy = j
                cx = i
                x = cx + 2*field[i][j][indx-1]
                y = cy + 2*field[i][j][indx]

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
            else:
                k += 1
    plt.figure(2)
    for i in range(len(keypoint)):

        plt.plot(dimentions[1]*keypoint[i][0], dimentions[0]-keypoint[i][1] *
                 dimentions[0], marker='o', color="black")
    plt.imshow(newImg)
    plt.show()
    raise ValueError


#----------------------#
# TRAINING UTILS
#----------------------#


def crop_on_mask(image, mask, threshold=0.6):
    # Crop the image around the predicted mask
    # Image: Tensor
    # Mask: Tensor (?)

    # rename to poseDataGenerator
    # Make a seperate function for visualizing the croped image and mask

    # TODO Optmize this function
    print(image.shape, type(image))
    print(mask.shape, type(mask))
    cropedImages = []
    cropedMask = []
    coordInfo = []
    for i in range(len(image)):
        coords = np.where(mask[i] >= threshold)[1:3]
        top_y = min(coords[0]) - 10
        top_x = min(coords[1]) - 10
        height = max(coords[0])-top_y + 20
        width = max(coords[1]) - top_x + 20

        info = [top_x, top_y, height, width]
        coordInfo.append(info)

        if not height % 2 == 0:
            height += 1
        if not width % 2 == 0:
            width += 1

        crop = torchvision.transforms.functional.crop(
            image[i], top_y, top_x, height, width)

        crop_mask = torchvision.transforms.functional.crop(
            mask[i], top_y, top_x, height, width)

        crop = torch.unsqueeze(crop, 0)
        crop_mask = torch.unsqueeze(crop_mask, 0)
        cropedImages.append(crop)
        cropedMask.append(crop_mask)
        print("Crop: ", crop.shape)
    # NB! Inverts the picutre colors (Might need to fix this)
    # print(cropedImages)
    #cropedImages = torch.cat(cropedImages, dim=0)
    # print(cropedImages.shape)
    crop = torch.squeeze(crop)
    crop_mask = torch.squeeze(crop_mask)
    crop = crop*255
    crop_mask = crop_mask*255
    print("MINIMASK: ", torch.unique(crop_mask))
    crop_mask_img = torchvision.transforms.ToPILImage()(crop_mask)
    img = torchvision.transforms.ToPILImage('RGB')(crop)
    img.show()
    crop_mask_img.show()

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
