from PIL import Image
import os, json, math, random, numpy as np, matplotlib as plt, cv2 as cv2


def filePathToArray(filePath, height = 600, width = 600): # uses PIL Image object to return image as numpy array
	image = Image.open(filePath)
	#image = image.resize((width, height))
	return np.array(image)


def formatStringToDict(string):
    newString = ""
    stringList = string.split(",")
    for i in range(len(stringList)):
        if i%2==0:
            newString += stringList[i]+"."
        else:
            newString += stringList[i]+","
    newString = newString[:-1]
    return json.loads(newString)

def getLabels(path):
	with open(path, encoding="utf-8") as f:
		#Labels er her all data som hentes fra .txt filen 
		labels = json.loads(f.readline())
		
	keyList = ["worldCornerCoordinates", "screenCornerCoordinates"]
	for key in keyList:
		labels[key] = formatStringToDict(labels[key])
	return labels

def getLabelsArray(dict):
	tmpCord = np.zeros((len(dict.keys()), 2))
	valuesList = list(dict.values())
	for i in range(len(valuesList)):
		tmpCord[i] = np.array([valuesList[i]["x"], valuesList[i]['y']])
	return tmpCord

def dictToArray(hypDict):
	coordArray = np.zeros((len(hypDict.keys()), 2))
	for key, hyps in hypDict.items():
		coordArray[key] = np.array([round(hyps[1]), round(hyps[0])]) # x, y format
	return coordArray



# takes input image and generates unit vector training data
def coordsTrainingGenerator(batchSize, masterList = None, height = 600, width = 600, altLabels = False): 
	
	basePath = '../../unity/My project/dataset/train/'
	if masterList == None:
		masterList = getMasterList(basePath)
		random.shuffle(masterList)
	i = 0
	#basePath = os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + model
	print(masterList)
	while True:
		xBatch = []
		yCoordBatch = []
		for b in range(1,batchSize):
			
			print(masterList[i])
			
			
			if i == len(masterList):
				i = 0
				random.shuffle(masterList)

			#Get images 
			x = filePathToArray(basePath + masterList[i][0], height, width)
			
			# altLabels satt til False  
			labels = getLabels(basePath + ('\\altLabels\\' if altLabels else masterList[i][2]))
			screenLabels = labels["screenCornerCoordinates"]

			#Number of corners in screen
			numCords = len(screenLabels)

			#How many coords? Avhenger av ant punkter 
			yCoordsLabels = np.zeros((height, width, 2*numCords)) # 9 coordinates, y,x for each 
			
			#Hent ID bilder 
			modelMask = filePathToArray(basePath + masterList[i][1], height, width)
			#Creates a list of unique RGB values in image
			#np.unique(modelMask.reshape(-1, modelMask.shape[2]), axis=0)
			
			
			for k in range(modelMask.shape[0]):
				for j in range(modelMask.shape[1]):

					if (modelMask[k][j]==np.array([0,0,0])).all():
						continue
					else:
						modelMask[k][j] = np.array([255,255,255])

			modelCoords = np.where(modelMask == 255)[:2]

			#y,x format 
			for modelCoord in zip(modelCoords[0][::3], modelCoords[1][::3]):
				setTrainingPixel(yCoordsLabels, modelCoord[0], modelCoord[1], screenLabels, height, width)
			
			xBatch.append(x)
			yCoordBatch.append(yCoordsLabels)
			i += 1
		yield (np.array(xBatch), np.array(yCoordBatch)) #yield?

def getMasterList(basePath): # returns list with image, mask, and label filenames
	totalList = sorted(os.listdir(basePath), key=len)
	imageList = list(filter(lambda x:"_img" in x, totalList))
	maskList = list(filter(lambda x:"_id" in x, totalList))
	labelList = list(filter(lambda x:".txt" in x, totalList))
	if len(imageList) != len(maskList) or len(imageList) != len(labelList):
		raise Exception("image, mask, and label list lengths do not match.")
	
	return [[a, b, c] for a, b, c in zip(imageList, maskList, labelList)]


def setTrainingPixel(outImage, y, x, screenLabels, height, width): # for each pixel given, calculate unit vectors to keypoints and store on pixel in outImage object
	
	for i in range(len(screenLabels.keys())):
		key = list(screenLabels.keys())[i]

		#yDiff = height * float(labels[i * 2 + 1]) - y # positive means y is above target in image
		yDiff = height * screenLabels[key]["y"] - y
		#xDiff = width * float(labels[i * 2]) - x # positive means x is left of target in image
		xDiff = height * screenLabels[key]["x"] - x

		mag = math.sqrt(yDiff ** 2 + xDiff ** 2)
	
		outImage[y][x][i * 2 + 1] = yDiff / mag # assign unit vectors pointing from coordinate to keypoint
		outImage[y][x][i * 2] = xDiff / mag




def pnp(p3d, p2d, drawPoints, matrix = np.array([[572.4114, 0., 325.2611], [0., 573.57043, 242.04899], [0., 0., 1.]]), method = cv2.SOLVEPNP_ITERATIVE):

	assert p3d.shape[0] == p2d.shape[0], 'points 3D and points 2D must have same number of vertices'

	p2d = np.ascontiguousarray(p2d.astype(np.float64))
	p3d = np.ascontiguousarray(p3d.astype(np.float64))
	matrix = matrix.astype(np.float64)
	try:
		_, R_exp, tVec = cv2.solvePnP(p3d,
								p2d,
								matrix,
								np.zeros(shape=[8, 1], dtype='float64'),
								flags=method)
	
	except Exception as e:
		print(e)
		#set_trace()
		print(p2d)

	#R_exp, t, _ = cv2.solvePnPRansac(p3d,
	#							p2d,
	#							matrix,
	#							distCoeffs,
	#							reprojectionError=12.0)

	#R, _ = cv2.Rodrigues(R_exp)
	
	(plotPoints, jacobian) = cv2.projectPoints(drawPoints, R_exp, tVec, matrix, np.zeros(shape=[8, 1], dtype='float64'))
	
	# return np.concatenate([R, tVec], axis=-1)
	return np.squeeze(plotPoints)


#Classes er mask beregnet av nettverk 
maskThreshold = 0.9

numHypotheses = 10
ransacThreshold = .99

points_3D = np.array([

		(0, 0, 0), #fbl
        (0, 0, -3), #fbr
        (0, 3, 0),  #ftl
        (0, 3, -3), #ftr
        (9, 0, 0), #bbl
        (9, 0, -3), #bbr
        (9, 3, 0), #btl
        (9 , 3, -3), #btr
        (4.5,1.5,-1.5) #Center 
                     ])

def predictPose(coords, classes, labels = False, checkPreds = False, altLabels = True):

	population = np.where(classes > maskThreshold)[:2]
	if not len(population):
		return False, None
	
	population = list(zip(population[0], population[1])) # y, x format
	
	hypDict = ransacVoting(population, coords)
 
	meanDict = getMean(hypDict)
	
	#covarDict = getCovariance(hypDict, meanDict)
	
	#
	pts3d = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + modelName + '\\', 'altPoints.txt'))
	
	preds = dictToArray(meanDict)

	'''
	if checkPreds is not False: # show predicted keypoints on image
		labelList = data.labelFloatsToPixels(labels)
		for ind in range(len(preds)):
			px = labelList[ind + 1][0]
			py = labelList[ind + 1][1]
			print("keypoint at " + str((px, py)))
			temp = np.array(checkPreds[py][px])
			checkPreds[py][px] = np.array([0,0,0])
			#plt.figure()
			#imshow(np.squeeze(checkPreds))
			#plt.show()
			checkPreds[py][px] = temp
	'''
	drawPoints = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + modelName + '\\', 'bb8_3d.txt'))
	
	return True, pnp(pts3d, preds, drawPoints)



def ransacVoting(population, coords): # ransac voting to generate 2d keypoint hypotheses
	hypDict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
	for n in range(numHypotheses): #take two pixels, find intersection of unit vectors
		#print(n)
		p1 = population.pop(random.randrange(len(population)))
		v1 = coords[p1[0]][p1[1]]
		p2 = population.pop(random.randrange(len(population)))
		v2 = coords[p2[0]][p2[1]]
		#print(p1, p2)
		#print(v1, v2)
		for i in range(9): # find lines intersection, use as hypothesis
			m1 = v1[i * 2 + 1] / v1[i * 2] # get slopes
			m2 = v2[i * 2 + 1] / v2[i * 2]
			if not (m1 - m2): # lines must intersect
				print('slope cancel')
				continue
			b1 = p1[0] - p1[1] * m1 # get y intercepts
			b2 = p2[0] - p2[1] * m2
			x = (b2 - b1) / (m1 - m2)
			y = m1 * x + b1
			
			weight = 0
			for voter in population: # voting for fit of hypothesis
				yDiff = y - voter[0]
				xDiff = x - voter[1]
				
				mag = math.sqrt(yDiff ** 2 + xDiff ** 2)
				vec = coords[voter[0]][voter[1]][i * 2: i * 2 + 2]
				
				if ransacVal(yDiff / mag, xDiff / mag, vec) > ransacThreshold:
					weight += 1
			hypDict[i].append(((y, x), weight))
			
		population.append(p1)
		population.append(p2)

		
	return hypDict

	
def ransacVal(y1, x1, v2): # dot product of unit vectors to find cos(theta difference)
	v2 = v2 / np.linalg.norm(v2)
	
	return y1 * v2[1] + x1 * v2[0]

def getMean(hypDict): # get weighted average of coordinates, weights list
	meanDict = {}
	for key, hyps in hypDict.items():
		xMean = 0
		yMean = 0
		totalWeight = 0
		for hyp in hyps:
			yMean += hyp[0][0] * hyp[1]
			xMean += hyp[0][1] * hyp[1]
			totalWeight += hyp[1]
		yMean /= totalWeight
		xMean /= totalWeight
		meanDict[key] = [yMean, xMean]
	return meanDict
