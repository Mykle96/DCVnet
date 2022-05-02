import torch
import math as m
import numpy as np
import cython

DEFAULT_CLASS = "container"


class VectorField:

    def __init__(self, cropedPredMask, cropedImage, classnames=DEFAULT_CLASS, keypoints=None):

        # Takes in a croped image of the container in focus. Which is found by the mask generated from the segmentation network
        # Creates a vector field for each keypoint (should be 9 keypoints in total for each container)
        # The vector-field is composed of unit direction vectors pointing from a pixel to a certain keypoint.
        """
        Args:
            cropedMask:
            cropedImage:
            classnames:
            keypoints: List of keypoints for the respective image and mask, format: [[x,y],[x2,y2],...,[x9,y9]]
        """
        self.cropedPredMask = cropedPredMask
        self.cropedImage = cropedImage
        self.classnames = classnames
        self.keypoints = keypoints
        numKeypoints = len(keypoints)

        assert keypoints is not None, f"A list of keypoints are need for generating a vector field! Please ensure the dataset contains keypoints or disable pose estimation."

        if numKeypoints < 8 & classnames:
            print(
                f"!!! An insufficent amount of keypoints,({numKeypoints}), detected for class {DEFAULT_CLASS}, this may have an impact on the final loss of the model if the number is not intentional.")
        elif numKeypoints > 10 & classnames:
            print(
                f"!!! An abundance of keypoints,({numKeypoints}), detected for class {DEFAULT_CLASS}, if not intentional this may cause obscurity in the final pose estimation.")
        else:
            print(f" {numKeypoints} keypoints registrered, for the image")

        dimentions = [cropedImage.shape[0],
                      cropedImage.shape[1]]  # [height, width]

        self.calculate_vector_field(
            cropedPredMask, cropedImage, keypoints, dimentions)

    def calculate_vector_field(self, cropedPredMask, cropedImage, keypoints, imgDimentions):
        """
        Function for calculating the unit direction vector field given the mask and image.
        This serves as the ground truth for the network to
        """
        # generate local variables
        image = []
        vectorField = []

        # Run through the croped image, where the mask is located (use only pixels located on the estimated mask)

        # generate a list for holding the vectors
        unitVectors = np.zeros((imgDimentions[0], imgDimentions[1], 18))
        # Get the mask coordinates from the croped mask image
        predMask = np.where(cropedPredMask == 255)[:2]
        # for each pixel in the mask, calculate the unit direction vector towards a keypoint
        for coordinates in zip(predMask[0][::3], predMask[1][::3]):
            self.find_unit_vector(unitVectors, coordinates,
                                  keypoints, imgDimentions)
        image.append(cropedImage)
        vectorField.append(unitVectors)
        # return a list of the image and the corresponding vector field
        return (np.array(image), np.array(vectorField))

    def find_unit_vector(self, vectors, pixel, keypoints, imgDimentions):
        # Calculates the unit vector between a given pixel in the mask and the respective keypoint
        # Pixel in the form [x,y]
        # Keypoint in the form [x,y]

        for keypoint in keypoints:
            # TODO Double check this loop, dont think it is quite right
            yDiff = imgDimentions[0]*float(keypoint*2+1) - pixel[1]
            xDiff = imgDimentions[1]*float(keypoint*2) - pixel[0]

            magnitude = m.sqrt(yDiff**2 + xDiff ** 2)
            vectors[pixel[0]][pixel[1]][keypoint*2+1] = yDiff/magnitude
            vectors[pixel[0]][pixel[1]][keypoint*2] = xDiff/magnitude
