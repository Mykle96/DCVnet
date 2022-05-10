import torch
#import engine
from torch.utils.data import DataLoader
from dataLoader import(SINTEFDataset)
from systemConfig import systems_configurations
from engine import *


# Hyperparameters
LEARNING_RATE = 0.004
BATCH_SIZE = 2
NUM_EPOCHS = 2
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = '../data/dummy-data/train'
TRAIN_MASK_DIR = '../data/dummy-data/train-mask'
VAL_IMG_DIR = '../data/dummy-data/validation'
VAL_MASK_DIR = '../data/dummy-data/valid-mask'
TEST_DIR = '../data/dummy-data/test'
TRAIN_DIR = '../data/dataset'
# DEVICE = systemConfig.systems_configurations()  # Sets the device to cpu or available gpu(s)

# Test load of dataset
"""
training_data = SINTEFDataset(imageDir=TRAIN_IMG_DIR,
                                maskDir=TRAIN_MASK_DIR, transform=None)
validation_data = SINTEFDataset(imageDir=VAL_IMG_DIR,
                                  maskDir=VAL_MASK_DIR,
                                  transform=None)

print(type(training_data))
print(len(training_data))
print(type(validation_data))
print(len(validation_data))

loaded_training_data = DataLoader(
    training_data, batch_size=BATCH_SIZE, shuffle=True)

loaded_validation_data = DataLoader(
    validation_data, batch_size=BATCH_SIZE, shuffle=True)

"""
classes = ["container"]
#test = SINTEFDataset(TRAIN_DIR, True)
test = SINTEFDataset(TRAIN_DIR, True)

loaded_test = DataLoader(test, batch_size=5, shuffle=True)

# print(loaded_test)
# print(type(loaded_test))

#test_f, test_l = next(iter(loaded_test))

unet = Model(classes=classes, pose_estimation=True)
losses = unet.train(loaded_test, None, epochs=11)
