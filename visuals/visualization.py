from matplotlib import pyplot as plt
import numpy
#from cv2 import cv2
import seaborn
import sys
modulename = "cv2"

assert modulename not in sys.modules, "{} was not properly imported! Make sure it is installed".format(
    modulename)
