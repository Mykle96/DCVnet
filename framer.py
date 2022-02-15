import cv2
import math
import matplotlib
import numpy
import time
import tqdm
import sys
import os


# This script extracts frames from videos and stores them as images.
# The amount frames extraceted varies with the frame rate of the video.
opencv = "cv2"
#assert opencv not in sys.modules, "{} was not properly imported! Make sure it is installed with {}".format(
   # opencv, "opencv-python")


def frame_maker(videoname: str, newDirectory: str, adjConst: float):
    """
    A function for generating images from frames of a given video. The function takes in 
    the name of a video located in the videos folder. It makes a new folder for the new
    extracted images in the raw-data folder.

    Args:
        videoname: The name of the video file (NB! Correct spelling or it will fail.)
        newDirectory: Name of the new directory to be created in the raw-data folder.

    return:
        Returns the frames of a video as images to a given path.
    """
    version = check_openCV_version()
    success = True
    currentFrame = 0
    imageNumber = 0

    videopath = "videos/{}".format(videoname)
    print("Extracting video from: {}".format(videopath))
    video = cv2.VideoCapture(videopath)
    fps = math.ceil(fps_checker(video, version))
    fps = fps*adjConst
    assert video.isOpened(
    ), "Something went wrong when opening {}, could not read the frame".format(videoname)
    outputPath = folder_generator(newDirectory)
    
    while success:
        success, frame = video.read()
        if currentFrame % fps == 0:
            if success:
               #cv2.imwrite((outputPath+newDirectory +
                cv2.imwrite((outputPath+"/"+
                            str(imageNumber)+".jpg"), frame)
            imageNumber += 1
        currentFrame += 1
    print("Finished running. {} images written to {} from video {}".format(imageNumber,outputPath,videoname))

def fps_checker(video, version):
    # Checks the frames per second (fps) in the video.
    
    assert type(version) == int, "Something went wrong with fetching the openCV version. Expected {}, but got {}".format(int(), type(version))
    if version < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print("FPS of video: {}".format(fps))
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        print("FPS of video: {}".format(fps))
    return fps


def check_openCV_version():
    # Checks the version of the OpenCV library currently used
    major_version = int(cv2.__version__.split(".")[0])
    return major_version


def folder_generator(newDirectory):
    outputPath = "raw-data/{}".format(newDirectory)
    assert os.path.isfile(outputPath) == False, "{} is already a folder under the raw-data folder. please change the name".format(newDirectory)
    os.mkdir(outputPath)
    return outputPath


#frame_maker("Loading and unloading of cargo in the port of Savannah, GA..mp4", "videodata",2)
#frame_maker("Container ship being loaded_unloaded time-lapse.mp4", "videodata2",0.2)
#frame_maker("Master.mp4", "videodata",1)
frame_maker("SUMMARY - Stinis Split-Headblock.mp4", "videodata",1)