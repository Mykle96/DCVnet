import torch
import numpy as np
import math
import tqdm
import os
from PIL import Image


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):

        super().__init__(dataset, collate_fn=DataLoader.collate_data, **kwargs)

    # Converts a list of tuples into a tuple of lists so that
    # it can properly be fed to the model for training
    @staticmethod
    def collate_data(batch):
        images, targets = zip(*batch)
        return list(images), list(targets)


class ShippingDataset(torch.utils.data.Dataset):
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
