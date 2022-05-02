import torch
from cv2 import cv2
import numpy as np
import math
import sys
import os
import torchvision
from dataLoader import ShippingDataset
from torch.utils.data import DataLoader


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


#----------------------#
# TRAINING UTILS
#----------------------#


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


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = ShippingDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = ShippingDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader
