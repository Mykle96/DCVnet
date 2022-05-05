import torch
import numpy as np
import math
import json
import os
from PIL import Image
from tqdm import tqdm

"""
class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):

        super().__init__(dataset, collate_fn=DataLoader.collate_data, **kwargs)

    # Converts a list of tuples into a tuple of lists so that
    # it can properly be fed to the model for training
    @staticmethod
    def collate_data(batch):
        images, targets = zip(*batch)
        return list(images), list(targets)
"""


class ShippingDataset(torch.utils.data.Dataset):
    def __init__(self, Dir, pose=False, transform=None):
        """
        A Class for initializing the SINTEF 6DPE ISO Container Dataset. It is important that images, masks and txt files
        are in the same directory, or else it will fail. The class formats the data like so: [[pose_index, maskIndex, imageIndex],...,]

        args:
            Dir: Path to the directory containing the dataset
            pose: Set to True if the pose data is to be used. (Default: False)
            transform: Specifiy custom transforms. Needs to be a list. (Default: None)

        return:
            None
        """
        self.basePath = Dir
        self.transform = transform
        self.pose = pose
        # List of all the images in the directory
        self.Dir = os.listdir(Dir)
        self.Dir.sort()
        sortedDir = self.Dir

        if pose:
            # If pose is set to True: check that there are enough meta files and update the local variables
            minFileThresh = len(sortedDir)/3
            txtFiles = len([f for f in os.listdir(Dir) if f.endswith('txt')])
            assert txtFiles == minFileThresh, f"There apears to be too few meta files in the the directory! Found {txtFiles}, but need {minFileThresh}"
            num_elements = 3
            pose_index = 0
            imageIndex = 2
            maskIndex = 1

        else:
            # if not pose, remove all excess txt files if there are any and update local variables
            sortedDir = [f for f in os.listdir(Dir) if f.endswith('png')]
            num_elements = 2
            imageIndex = 1
            maskIndex = 0
        # Split the dataset into lists of lists containing the pose, mask and image
        data = [sortedDir[x:x+num_elements]
                for x in range(0, len(sortedDir), num_elements)]
        print("Initializing the dataset")
        for index in tqdm(range(len(data))):
            data[index][imageIndex] = np.array(Image.open(
                self.basePath+"/"+data[index][imageIndex]).convert("RGB"))
            data[index][maskIndex] = np.array(Image.open(
                self.basePath+"/"+data[index][maskIndex]).convert("L"), dtype=np.float32)
            if pose:
                poseDict = self.getPoseData(
                    self.basePath+"/"+data[index][pose_index])
                labelsArray = self.getLabelsArray(
                    poseDict["screenCornerCoordinates"])
                data[index][pose_index] = labelsArray
        self.Dir = data
        return

    def __len__(self):
        return len(self.Dir)

    def __getitem__(self, index):
        # Needs to return a list with the image, mask and keypoints (if pose is true)
        if self.pose:
            image = self.Dir[index][2]
            mask = self.Dir[index][1]
            keypoints = self.Dir[index][0]
        else:
            image = self.Dir[index][1]
            mask = self.Dir[index][0]

        print("Mask shape: ", mask.shape)
        print("Image shape: ", image.shape)
        # Need to have default transformations if transformations are set to NONE
        if self.transform is not None:
            raise NotImplementedError(
                "Transformations are not handled yet! set to None")
        if self.pose:
            return keypoints, image, mask
        else:
            return image, mask

    def formatStringToDict(self, string):
        newString = ""
        stringList = string.split(",")
        for i in range(len(stringList)):
            if i % 2 == 0:
                newString += stringList[i]+"."
            else:
                newString += stringList[i]+","
        newString = newString[:-1]
        return json.loads(newString)

    def getPoseData(self, path):
        with open(path, encoding="utf-8") as f:
            # Labels er her all data som hentes fra .txt filen
            labels = json.loads(f.readline())
        keyList = ["worldCornerCoordinates", "screenCornerCoordinates"]
        for key in keyList:
            labels[key] = self.formatStringToDict(labels[key])
        return labels

    def getLabelsArray(self, dict):
        tmpCord = np.zeros((len(dict.keys()), 2))
        valuesList = list(dict.values())
        for i in range(len(valuesList)):
            tmpCord[i] = np.array([valuesList[i]["x"], valuesList[i]['y']])
        return tmpCord


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, imageDir, maskDir, transform=None):
        # data loading
        self.imageDir = imageDir
        self.maskDir = maskDir
        self.transform = transform
        # List of all the images in the directory
        self.images = os.listdir(imageDir)
        return

    def __len__(self):
        # length of dataset
        return len(self.images)

    def __getitem__(self, index):
        imgPath = os.path.join(self.imageDir, self.images[index])
        maskPath = os.path.join(self.maskDir, self.images[index])
        image = np.array(Image.open(imgPath).convert("RGB"))
        mask = np.array(Image.open(maskPath).convert("L"), dtype=np.float32)
        print(mask.shape)
        mask[mask == 255.0] = 1.0

        # Need to have default transformations if transformations are set to NONE
        if self.transform is not None:
            raise NotImplementedError(
                "Transformations are not handled yet! set to None")
