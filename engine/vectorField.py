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

        args:
            target: Mask tensor with shape [batch size, 1, dimensionX, dimensionY]
            image: Image tensor with shape [batch size, 3, dimensionX, dimensionY]
            keypoints: Keypoint tensor with shape [batch size, number of keypoints, keypoint coordinates]
        """
        # TODO: Does this class really need to be inizialized with the target, image and keypoints?
        self.target = target
        self.image = image
        self.classnames = classnames
        self.keypoints = keypoints
        numKeypoints = keypoints.shape[1]

        assert keypoints is not None, f"A list of keypoints are need for generating a vector field! Please ensure the dataset contains keypoints or disable pose estimation."

        if numKeypoints < 8:
            print(
                f"!!! An insufficent amount of keypoints,({numKeypoints}), detected for class {DEFAULT_CLASS}, this may have an impact on the final loss of the model if the number is not intentional.")
        elif numKeypoints > 10:
            print(
                f"!!! An abundance of keypoints,({numKeypoints}), detected for class {DEFAULT_CLASS}, if not intentional this may cause obscurity in the final pose estimation.")
        else:
            print("")
            print(f"---> {numKeypoints} keypoints registrered, for the image")
            print("")

    def calculate_vector_field(self, targets, images, keypoints, coordInfo):
        """
        Function for calculating the unit direction vector field given the mask and image.
        This serves as the ground truth for the network durning keypoint localization training.

        args:
            target: ground truth mask list with length=batch size on the format [tensors,.., tensor], 
            tensor.shape = (1,1,dimY,dimX)
            image: list of images with length=batch size on the format [tensors,.., tensor], 
            tensor.shape = (1,3,dimY,dimX)
            keypoints: Keypoint tensor with shape [batch size, number of keypoints, keypoint coordinates]
            coordInfo: List of lists with cropping coordinates, length=batch size, 
            on the format: [[top_x, top_y, height, width]]

        return: 
            returns a tuple with a arrays of vecotr fields on the format: 
                np.array(np.array([dimy,dimx,2*num keypoints]), np.array[,,]) 
            and a tensor with the corresponding transformed keypoints in screen coordinates 
            on the format [[x1,y1],..]
        """

        if type(keypoints) == list:
            pass
        elif type(keypoints) == dict:
            keypoints = list(keypoints.values())
        elif type(keypoints) == torch.Tensor:
            keypoints = keypoints.numpy()
        else:
            raise ValueError(
                f"Excpected type tensor, list or dict, but got {type(keypoints)}. calculate_vector_field function can only handle lists or dicts of keypoints.")

        if not (len(images) == len(targets) == keypoints.shape[0]):
            print(f"Number of images, masks and keypoints is not equal")
            print(
                f"Num images: {images.shape[0]}, num masks: {targets.shape[0]}, num keypoints: {keypoints.shape[0]}",)
            return False
        else:
            numImages = len(images)
            # Generating a tensor with new keypoints to cropped screen, [x,y] format
            new_keypoints = torch.tensor(
                self.update_keypoint(keypoints, coordInfo))

            vectorFieldList = []
            print("Calculating unit vector fields")
            for i in tqdm(range(numImages)):
                # generate local variables

                image = images[i].permute(0, 2, 3, 1).cpu().numpy()
                image = np.squeeze(image, axis=0)
                target = targets[i].permute(0, 2, 3, 1).cpu().numpy()
                target = np.squeeze(target, axis=0)

                numKeypoints = new_keypoints[i].shape[0]
                dimentions = [image.shape[0],
                              image.shape[1]]  # [height, width]

                # generate a array for holding the vectors
                unitVectors = np.zeros(
                    (dimentions[0], dimentions[1], numKeypoints*2))

                # Get the mask coordinates from the mask image
                mask = np.where(target != 0)[:2]  # FORMAT: y,x

                # for each pixel in the mask, calculate the unit direction vector towards a keypoint
                for coordinates in zip(mask[0], mask[1]):
                    self.find_unit_vector(unitVectors, coordinates,
                                          new_keypoints[i], dimentions)

                vectorFieldList.append(unitVectors)

        return (np.array(vectorFieldList), new_keypoints)

    def find_unit_vector(self, vectors, pixel, keypoints, imgDimentions):
        """
        Function for calculating the unit direction vector between a given pixel and all keypoints given. The function
        updates a list of vectors with each calculated unit vector.

        args:
            vectors: array of unit vectors to be updated [dimy, dimx, number of keypoints*2]
            pixel: the current pixel [y,x] 
            keypoints: the current keypoint [[x1,y1],[x2,y2]...]
            imgDimentions: the diemntions of the input image [y,x]

        returns:
            No return
        """

        for index, keypoint in enumerate(keypoints):

            yDiff = imgDimentions[0]*float(1-keypoint[1]) - pixel[0]
            xDiff = imgDimentions[1]*float(keypoint[0]) - pixel[1]

            magnitude = m.sqrt(yDiff**2 + xDiff ** 2)

            # Unit vectors on the format [x1, y1, ....xn, yn]  for each  pixel with n keypoints
            vectors[pixel[0]][pixel[1]][index*2+1] = yDiff/magnitude
            vectors[pixel[0]][pixel[1]][index*2] = xDiff/magnitude

    def visualize_gt_vectorfield(self, field, keypoint, indx=5, imgIndx=-1, oneImage=False, saveImages=False):
        '''
        Function to visualize vector field towards a certain keypoint, and plotting all keypoint

        args:
            field:  Arrays with len=batch_size, with vector fields on the format: 
                np.array(np.array([dimy,dimx,2*num keypoints]), np.array[,,])
            keypoint: Tensor with the format (batch size, number of keypoints, 2)
            indx: keypoint index, default is last keypoint
            imgIndx: batch index of image, default is first image in batch 

        returns:
            No return
        '''

        if not isinstance(field, np.ndarray):
            field = field.numpy()
        keypoint = keypoint.numpy()

        # Get vector field and keypoints for a specfic image
        if (imgIndx > len(field)-1 or imgIndx < 0):
            print(
                f"Image index = {imgIndx} outside of interval [0, {len(field)-1}]")
            print("Setting image index to the last element")
            imgIndx = -1
        field = field[imgIndx]
        all_keypoints = keypoint[imgIndx]

        dimensions = [field.shape[0], field.shape[1]]  # y,x
        numCords = int(field.shape[2]/2)

        if not(oneImage):
            rows = m.ceil(m.sqrt(numCords))
            cols = m.ceil(m.sqrt(numCords))
            figg, ax = plt.subplots(
                rows, cols, figsize=(10, 10))  # ax[y,x] i plottet
            ax = ax.flatten()
            numImages = numCords
        else:
            numImages = 1
            if(indx == -1):
                indx = numCords-1
            elif(indx > numCords or indx < 0):
                raise ValueError(
                    f"Keypoint value = {indx} needs to be in the interval [1, number of keypoints = {numCords}]")

        for index in range(numImages):
            newImg = np.zeros((dimensions[0], dimensions[1], 3))
            if(oneImage):
                index = indx
            for i in range(dimensions[0]):
                for j in range(dimensions[1]):
                    if(field[i][j] != np.zeros(2*numCords)).all():

                        cy = j
                        cx = i
                        x = cx + 2*field[i][j][2*index]
                        y = cy + 2*field[i][j][2*index+1]

                        if(cx-x) < 0:
                            # (2) og (3)
                            angle = m.atan((cy-y)/(cx-x))+m.pi
                        elif(cy-y) < 0:
                            # (4)
                            if(cx == x):
                                # 270 degrees
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

            if not(oneImage):
                ax[index].imshow(newImg)

            for k in range(len(all_keypoints)):
                if k == index:
                    marker = '+'
                    color = 'white'
                else:
                    marker = '.'
                    color = 'white'

                if not(oneImage):
                    ax[index].plot(dimensions[1]*all_keypoints[k][0], dimensions[0]-all_keypoints[k][1] *
                                   dimensions[0], marker=marker, color=color)
                else:
                    plt.plot(dimensions[1]*all_keypoints[k][0], dimensions[0]-all_keypoints[k][1] *
                             dimensions[0], marker=marker, color='white')
        if(oneImage):
            plt.imshow(newImg)
        if(saveImages):
            pass
        else:
            plt.show()

    def update_keypoint(self, keypoints, coordsInfo):
        updated_keypoints = []
        '''
        Keypoints type = np.ndarray with shape [[[x1,y1],..,[xn,yn]],...] 
        Converting screen coordinates for consistency, by dividing on width/height of cropped screen
        coordInfo: List of lists on the format: [[top_x, top_y, height, width],...,[..]]

        returns a list with screencoordinates to new screen on the format [[x1,y1],[x2,y2],...,]
        '''

        for index, keypoint in enumerate(keypoints):
            loop = []
            # keypoint is now a list of lists with x,y screencoordinates of keypoints
            for element in keypoint:

                updated_keypoint_x = float((
                    600*element[0]-coordsInfo[index][0])/coordsInfo[index][3])
                updated_keypoint_y = float((
                    600*(element[1]-1)+coordsInfo[index][1]+coordsInfo[index][2])/coordsInfo[index][2])

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
