import torch
import numpy as np
import math
import json
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


class SINTEFDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):

        super().__init__(dataset, collate_fn=SINTEFDataLoader.collate_data, **kwargs)
    # NOT USED

    @staticmethod
    def collate_data(batch):
        images, targets = zip(*batch)


class SINTEFDataset(torch.utils.data.Dataset):
    def __init__(self, Dir, pose=False, transform=None):
        """
        A Class for initializing the SINTEF 6DPE ISO Container Dataset. It is important that images, masks and txt files
        are in the same directory, or else it will fail. The class formats the data like so: [[imageIndex, maskIndex, pose_index],...,]

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
        data = self.getDataList(Dir)
        if pose:
            # If pose is set to True: check that there are enough meta files and update the local variables
            num_elements = 3
            pose_index = 2
            imageIndex = 0
            maskIndex = 1

        else:
            # if not pose, remove all excess txt files if there are any and update local variables
            num_elements = 2
            imageIndex = 0
            maskIndex = 1

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
            image = self.Dir[index][0]
            mask = self.Dir[index][1]
            mask[mask != 0.0] = 1.0
            keypoints = self.Dir[index][2]
        else:
            image = self.Dir[index][0]
            mask = self.Dir[index][1]
            mask[mask != 0.0] = 1.0
        #mask[mask != 0] = 255
        # Need to have default transformations if transformations are set to NONE
        if self.transform is not None:
            raise NotImplementedError(
                "Custom Transformations are not handled yet! set to None")
        else:
            # Check if the images are of the dimentions of 600x600, if not set them to that size
            pass

        if self.pose:
            return image, mask, keypoints
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

    def getDataList(self, dirPath):
        """
        This monstrosity of a function returns the data in a list format of [[image, mask, meta_file],...,]
        It splits the dataset into three (if pose is selected) lists, sorts them in ascending order and combines
        them agin.
        """
        dataList = sorted(os.listdir(dirPath), key=len)
        if self.pose:

            # Split up the image files and sort in ascending order
            images = list(filter(lambda x: "_img" in x, dataList))
            images = sorted(
                images, key=lambda x: int(x.split('_', 1)[0]))

            # Split up the mask files and sort in ascending order
            masks = list(filter(lambda x: "_id" in x, dataList))
            masks = sorted(masks, key=lambda x: int(x.split('_', 1)[0]))

            # Split up the label files and sort in ascending order
            labels = list(filter(lambda x: ".txt" in x, dataList))
            labels = sorted(
                labels, key=lambda x: int(x.split('.', 1)[0]))

            numImages = len(images)
            numLabels = len(labels)
            assert numImages == numLabels, f"The number of label files does not match the number of images! Images: {numImages}, label files: {numLabels}"

            return [[a, b, c] for a, b, c in zip(images, masks, labels)]
        else:
            # If not pose, remove all the txt files
            self.Dir = [f for f in dataList if f.endswith('png')]

            # Split up the image files and sort in ascending order
            images = list(filter(lambda x: "_img" in x, dataList))
            images = sorted(
                images, key=lambda x: int(x.split('_', 1)[0]))

            # Split up the mask files and sort in ascending order
            masks = list(filter(lambda x: "_id" in x,    dataList))
            masks = sorted(masks, key=lambda x: int(x.split('_', 1)[0]))

            numImages = len(images)
            numMasks = len(masks)
            assert numImages == numMasks, f"The number of masks does not match the number of images! Images: {numImages}, Masks: {numMasks}"

            return [[a, b] for a, b in zip(images, masks)]


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
