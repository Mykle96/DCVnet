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

    def calculate_vector_field(self, targets, images, keypoints, coordInfo):
        """
        Function for calculating the unit direction vector field given the mask and image.
        This serves as the ground truth for the network durning keypoint localization training.

        args:
            target: the ground truth mask (note, only one mask can be served)
            image: the corresponding image of the mask
            keypoints: list of keypoints of the object in the image

        return:
            returns a tensor containing unit vector field corresponding to all images in a batch 
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

        print(len(images))
        print(len(targets))
        print(images[0].shape)
        print(targets[0].shape)
        keypoints = keypoints.numpy()

        if not (len(images) == len(targets) == keypoints.shape[0]):
            print(f"Number of images, masks and keypoints is not equal")
            print(
                f"Num images: {images.shape[0]}, num masks: {targets.shape[0]}, num keypoints: {keypoints.shape[0]}",)
            return False
        else:
            numImages = len(images)
            # new keypoints
            new_keypoints = torch.tensor(
                self.update_keypoint(keypoints, coordInfo))
            # imageList = []
            print("New: ", new_keypoints.shape)
            vectorFieldList = []
            print("Calculating unit vector fields")
            for i in tqdm(range(numImages)):
                # generate local variables

                image = images[i].permute(0, 2, 3, 1).numpy()
                image = np.squeeze(image, axis=0)
                target = targets[i].permute(0, 2, 3, 1).numpy()
                target = np.squeeze(target, axis=0)
                print("Inside loop ", image.shape)
                print("Inside loop ", target.shape)

                numKeypoints = new_keypoints[i].shape[0]
                dimentions = [image.shape[0],
                              image.shape[1]]
                print("dim ", dimentions)
                # [height, width]
                # generate a array for holding the vectors

                unitVectors = np.zeros(
                    (dimentions[0], dimentions[1], numKeypoints*2))

                # Get the mask coordinates from the mask image
                mask = np.where(target != 0)[:2]

                # for each pixel in the mask, calculate the unit direction vector towards a keypoint
                for coordinates in zip(mask[0], mask[1]):
                    self.find_unit_vector(unitVectors, coordinates,
                                          new_keypoints[i], dimentions)
                # imageList.append(images[i])
                vectorFieldList.append(unitVectors)

            # return a tuple of the image and the corresponding vector field

        return (np.array(vectorFieldList), new_keypoints)

    def find_unit_vector(self, vectors, pixel, keypoints, imgDimentions):
        """
        Function for calculating the unit direction vector between a given pixel and the respective keypoint. The function
        updates a list of vectors with each calculated unit vector.

        args:
            vectors: array of unit vectors to be updated
            pixel: the current pixel [y,x]
            keypoints: the current keypoint [x,y]
            imgDimentions: the diemntions of the input image [y,x]

        returns:
            No return
        """

        for index, keypoint in enumerate(keypoints):
            # TODO Double check this loop, dont think it is quite right
            yDiff = imgDimentions[0]*float(1-keypoint[1]) - pixel[0]
            xDiff = imgDimentions[1]*float(keypoint[0]) - pixel[1]

            magnitude = m.sqrt(yDiff**2 + xDiff ** 2)

            # vectors[pixel[0]][pixel[1]][keypoint[1]*2+1] = yDiff/magnitude
            # vectors on the format [x1, y1, ....xn, yn]  for each  pixel
            vectors[pixel[0]][pixel[1]][index*2+1] = yDiff/magnitude
            vectors[pixel[0]][pixel[1]][index*2] = xDiff/magnitude

    def visualize_gt_vectorfield(self, field, keypoint, indx=-1):
        # Takes in a np_array of the found unit vector field and displays it

        # trainPoseData [5,600,600,18] tensor
        # keypoint (5,9,2) tensor
        # imgInt = random.randint(0, field.shape[0]-1)
        imgInt = -1
        if not isinstance(field, np.ndarray):
            field = field.numpy()
        keypoint = keypoint.numpy()
        print("fieldshape :", field.shape)
        field = field[imgInt]
        print("SHAPE: ", field.shape)
        all_keypoints = keypoint[imgInt]
        print("Keypoints: ", all_keypoints.shape)
        keypoint = keypoint[imgInt][indx]
        dimensions = [field.shape[0], field.shape[1]]  # y,x
        print("DIMENTIONS: ", dimensions)

        newImg = np.zeros((dimensions[0], dimensions[1], 3))
        numCords = int(field.shape[2]/2)

        k = 0
        for i in range(dimensions[0]):  # 90
            for j in range(dimensions[1]):
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
        plt.figure(1)
        for i in range(len(all_keypoints)):

            plt.plot(dimensions[1]*all_keypoints[i][0], dimensions[0]-all_keypoints[i][1] *
                     dimensions[0], marker='.', color="white")
        plt.imshow(newImg)
        plt.show()

    def update_keypoint(self, keypoints, coordsInfo):
        updated_keypoints = []
        # Keypoints shape: [[x,y]]
        # converting pixel coordinates for consistency, by dividing on width/height of cropped screen
        # new_X = (600*old_x - cx)/new_width
        # new_Y = (600(old_y -1)+ cy + new_height)/new_height
        # print("Keypoints: ", keypoints)

        for index, keypoint in enumerate(keypoints):
            loop = []

            for element in keypoint:
                #print("ELEM ", element)
                updated_keypoint_x = float((
                    600*element[0]-coordsInfo[index][0])/coordsInfo[index][3])
                updated_keypoint_y = float((
                    600*(element[1]-1)+coordsInfo[index][1]+coordsInfo[index][2])/coordsInfo[index][2])
                #print(updated_keypoint_x, updated_keypoint_y)
                loop.append(
                    [updated_keypoint_x, updated_keypoint_y])

            updated_keypoints.append(loop)

        return updated_keypoints


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
