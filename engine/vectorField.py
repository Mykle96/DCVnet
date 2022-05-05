import torch
import math as m
import numpy as np
import matplotlib.pyplot as plt
import cython

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

    def calculate_vector_field(self, target, image, keypoints):
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

        else:
            raise ValueError(
                f"Excpected type list or dict, but got {type(keypoints)}. calculate_vector_field function can only handle lists or dicts of keypoints.")

        # generate local variables
        images = []
        vectorField = []
        dimentions = [image.shape[0],
                      image.shape[1]]  # [height, width]
        # generate a array for holding the vectors

        unitVectors = np.zeros(
            (dimentions[0], dimentions[1], len(keypoints)*2))
        # Get the mask coordinates from the mask image
        mask = np.where(target == 255)[:2]
        # for each pixel in the mask, calculate the unit direction vector towards a keypoint
        for coordinates in zip(mask[0][::3], mask[1][::3]):
            self.find_unit_vector(unitVectors, coordinates,
                                  keypoints, dimentions)
        images.append(image)
        vectorField.append(unitVectors)
        # return a tuple of the image and the corresponding vector field

        return (np.array(images), np.array(vectorField))

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

        for keypoint in keypoints:
            # TODO Double check this loop, dont think it is quite right
            yDiff = imgDimentions[0]*float(keypoint*2+1) - pixel[1]
            xDiff = imgDimentions[1]*float(keypoint*2) - pixel[0]

            magnitude = m.sqrt(yDiff**2 + xDiff ** 2)
            vectors[pixel[0]][pixel[1]][keypoint*2+1] = yDiff/magnitude
            vectors[pixel[0]][pixel[1]][keypoint*2] = xDiff/magnitude

    def visualize_gt_vectorfield(self, field):
        # Takes in a np_array of the found unit vector field and displays it
        raise NotImplementedError()


def test():
    # Test of vector field
    imagePath = "../data/dummy-data/train/1_img.png"
    maskPath = "../data/dummy-data/train-mask/1_id.png"
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
