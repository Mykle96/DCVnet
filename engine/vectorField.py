import torch
import math as m
import numpy as np
import matplotlib.pyplot as plt
import cython
import colorsys
from tqdm import tqdm
import random


DEFAULT_CLASS = "container"


class VectorField:

    def __init__(self, target, image, keypoints, classnames=DEFAULT_CLASS):
        """
        Class for generating a vector field of unit direction vectors from an arbitrary pixel to a respectiv keypoint
        located on a given object. Creates a vector field for each keypoint.

        """
        # TODO: Does this class really need to be inizialized with the target, image and keypoints?
        self.target = target
        self.image = image
        self.classnames = classnames
        self.keypoints = keypoints
        numKeypoints = len(keypoints)

        assert keypoints is not None, f"A list of keypoints are need for generating a vector field! Please ensure the dataset contains keypoints or disable pose estimation."

        if numKeypoints < 8:
            print(
                f"!!! An insufficent amount of keypoints,({numKeypoints}), detected for class {DEFAULT_CLASS}, this may have an impact on the final loss of the model if the number is not intentional.")
        elif numKeypoints > 10:
            print(
                f"!!! An abundance of keypoints,({numKeypoints}), detected for class {DEFAULT_CLASS}, if not intentional this may cause obscurity in the final pose estimation.")
        else:
            print(f" {numKeypoints} keypoints registrered, for the image")

    def calculate_vector_field(self, targets, images, keypoints):
        """
        Function for calculating the unit direction vector field given the mask and image.
        This serves as the ground truth for the network durning keypoint localization training.

        args:
            target: the ground truth mask (note, only one mask can be served)
            image: the corresponding image of the mask
            keypoints: list of keypoints of the object in the image

        return:
            returns a tuple containg the image and the corresponding unit vector field in the same dimentions as the image
        """

        if type(keypoints) == list:
            pass

        elif type(keypoints) == dict:
            keypoints = list(keypoints.values())
        elif type(keypoints) == torch.Tensor:
            keypoints.tolist()
        else:
            raise ValueError(
                f"Excpected type list or dict, but got {type(keypoints)}. calculate_vector_field function can only handle lists or dicts of keypoints.")

        images = images.permute(0, 2, 3, 1).numpy()
        targets = targets.permute(0, 2, 3, 1).numpy()
        keypoints = keypoints.numpy()

        if not (images.shape[0] == targets.shape[0] == keypoints.shape[0]):
            print(f"Number of images, masks and keypoints is not equal")
            print(
                f"Num images: {images.shape[0]}, num masks: {targets.shape[0]}, num keypoints: {keypoints.shape[0]}",)
            return False
        else:
            numImages = images.shape[0]

            #imageList = []
            vectorFieldList = []
            print("Calculating unit vector fields")
            for i in tqdm(range(numImages)):
                # generate local variables
                numKeypoints = keypoints[i].shape[0]
                dimentions = [images[i].shape[0],
                              images[i].shape[1]]
                # [height, width]
                # generate a array for holding the vectors

                unitVectors = np.zeros(
                    (dimentions[0], dimentions[1], numKeypoints*2))

                # Get the mask coordinates from the mask image
                mask = np.where(targets[i] != 0)[:2]

                # for each pixel in the mask, calculate the unit direction vector towards a keypoint
                for coordinates in zip(mask[0], mask[1]):
                    self.find_unit_vector(unitVectors, coordinates,
                                          keypoints[i], dimentions)
                # imageList.append(images[i])
                vectorFieldList.append(unitVectors)

            # return a tuple of the image and the corresponding vector field

        return torch.tensor(np.array(vectorFieldList))

    def find_unit_vector(self, vectors, pixel, keypoints, imgDimentions):
        """
        Function for calculating the unit direction vector between a given pixel and the respective keypoint. The function 
        updates a list of vectors with each calculated unit vector.

        args:
            vectors: array of unit vectors to be updated
            pixel: the current pixel [x,y]
            keypoints: the current keypoint
            imgDimentions: the diemntions of the input image

        returns:
            No return
        """

        for index, keypoint in enumerate(keypoints):

            # TODO Double check this loop, dont think it is quite right
            yDiff = imgDimentions[0]*float(keypoint[1]) - pixel[1]
            xDiff = imgDimentions[1]*float(keypoint[0]) - pixel[0]

            magnitude = m.sqrt(yDiff**2 + xDiff ** 2)

            #vectors[pixel[0]][pixel[1]][keypoint[1]*2+1] = yDiff/magnitude
            vectors[pixel[0]][pixel[1]][index*2+1] = yDiff/magnitude
            vectors[pixel[0]][pixel[1]][index*2] = xDiff/magnitude

    def visualize_gt_vectorfield(self, field, keypoint, indx=-1):
        # Takes in a np_array of the found unit vector field and displays it
        # Keypoint is a single keypoint on the format (x,y)
        # Field is a vectorfield on the format (600,600,18)

        # trainPoseData [5,600,600,18] tensor
        # keypoint (5,9,2) tensor

        imgInt = random.randint(0, field.shape[0]-1)

        field = field.numpy()
        keypoint = keypoint.numpy()
        field = field[imgInt]
        keypoint = keypoint[imgInt][indx]

        newImg = np.zeros((600, 600, 3))
        numCords = int(field.shape[2]/2)

        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                if(field[i][j] != np.zeros(2*numCords)).all():
                    y = i
                    x = j
                    cx = int(round(keypoint[0]*600))
                    cy = int(round(600-keypoint[1]*600))
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

        plt.plot(cx, cy, marker='.', color="white")
        plt.imshow(newImg)
        plt.show()


def test():
    # Test of vector field
    imagePath = "../data/dummy-data/test/1_img.png"
    maskPath = "../data/dummy-data/test/1_id.png"
    keypoints = [[459, 240], [358, 207], [184, 455], [309, 509], [
        309, 576], [198, 525], [447, 313], [354, 289], [331, 369]]
    image = plt.imread(imagePath)
    mask = plt.imread(maskPath)
    indices = np.where(mask == 255)
    print(indices)
    print(mask)
    plt.imshow(image)
    # plt.show()
    plt.imshow(mask)
    # plt.show()
    field = VectorField(image, mask, keypoints)
    vectorfield = field.calculate_vector_field(image, mask, keypoints)
    print(vectorfield)


if __name__ == "__main__":
    test()
