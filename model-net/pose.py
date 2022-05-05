import secrets
from turtle import color
import cv2 
from PIL import Image
import data, numpy as np, matplotlib.pyplot as plt

numHypotheses = 50 #  hypotheses considered for each keypoint
ransacThreshold = .99 # min value for population sample to agree with proposed hypothesis
maskThreshold = .9 # min value for pixel to be included in population of ransac process
pruneBool = True # flag to perform pruning operation
pruneRatio = .5 # percent of smallest weighted hypotheses to be pruned
noiseScale = .1 # artificially add noise to true target data, used for testing pnp accuracy
minHyps = 55
checkQuads = True



#Brukes for å lage dict av predikerte koordinater 
def labelDrawPoints(drawList): # (b, f = back, front), (l, r = left, right), (u, d = up , down)
	drawDict = {}
	drawDict['bld'] = (int(round(drawList[0][0])), int(round(drawList[0][1])))
	drawDict['blu'] = (int(round(drawList[1][0])), int(round(drawList[1][1])))
	drawDict['fld'] = (int(round(drawList[2][0])), int(round(drawList[2][1])))
	drawDict['flu'] = (int(round(drawList[3][0])), int(round(drawList[3][1])))
	drawDict['brd'] = (int(round(drawList[4][0])), int(round(drawList[4][1])))
	drawDict['bru'] = (int(round(drawList[5][0])), int(round(drawList[5][1])))
	drawDict['frd'] = (int(round(drawList[6][0])), int(round(drawList[6][1])))
	drawDict['fru'] = (int(round(drawList[7][0])), int(round(drawList[7][1])))
	return drawDict

#(b, f = back, front), (l, r = left, right), (u, d = up , down)
def drawPose(img, drawPoints, colour = (255,0,0), gt = False, height=600, width=600): # draw bounding box
	
	if(gt): #If ground truth:
		for i in drawPoints:
			drawPoints[i] = [
				int(round(drawPoints[i]["x"]*width)),
				 600-int(round(drawPoints[i]["y"]*height))
				 ]
			print(drawPoints[i])

	cv2.line(img, drawPoints['fbl'], drawPoints['fbr'], colour, 1)
	cv2.line(img, drawPoints['fbl'], drawPoints['ftl'], colour, 1)
	cv2.line(img, drawPoints['ftl'], drawPoints['ftr'], colour, 1)
	cv2.line(img, drawPoints['fbr'], drawPoints['ftr'], colour, 1)
	cv2.line(img, drawPoints['ftr'], drawPoints['btr'], colour, 1)
	cv2.line(img, drawPoints['ftl'], drawPoints['btl'], colour, 1)
	cv2.line(img, drawPoints['fbl'], drawPoints['bbl'], colour, 1)
	cv2.line(img, drawPoints['fbr'], drawPoints['bbr'], colour, 1)
	cv2.line(img, drawPoints['bbl'], drawPoints['bbr'], colour, 1)
	cv2.line(img, drawPoints['bbl'], drawPoints['btl'], colour, 1)
	cv2.line(img, drawPoints['bbr'], drawPoints['btr'], colour, 1)
	cv2.line(img, drawPoints['btl'], drawPoints['btr'], colour, 1)
	



'''
labels = data.getLabels('../../unity/My project/dataset/train/1.txt')
screenLabels = labels["screenCornerCoordinates"]
print(screenLabels)
print(screenLabels['fbl'])


image = np.array(Image.open('../../unity/My project/dataset/train/1_img.png'))
drawPose(image, screenLabels, gt=True)
plt.imshow(image)

img = Image.fromarray(image, 'RGB')
img.show()
'''
