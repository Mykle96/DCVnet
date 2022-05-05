import data
import tensorflow as tf, numpy as np
from tensorflow.keras import backend as K


def stvNet(inputShape = (600, 600, 3), outVectors = True, outClasses = False, modelName = "stvNet"):
	
	xIn = tf.keras.Input(inputShape, dtype = np.dtype('uint8'))
	
	#Deler alle verdiene i bildet på 255? 
	x = tf.keras.layers.Lambda(lambda x: x / 255) (xIn)
	
	x = tf.keras.layers.Conv2D(64, 7, input_shape = inputShape, kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0), padding = 'same')(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation('relu')(x)
	
	res1 = x
	
	x = tf.keras.layers.MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(x)

	skip = x
	
	x = convLayer(x, 64, 3)
	x = convLayer(x, 64, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = x
	
	x = convLayer(x, 64, 3)
	x = convLayer(x, 64, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = tf.keras.layers.MaxPool2D(pool_size = 2, padding = 'same')(x)
	skip = tf.pad(skip, [[0, 0], [0, 0], [0, 0], [32, 32]]) # linear projection
	res2 = x
	
	x = convLayer(x, 128, 3, 2)
	x = convLayer(x, 128, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = x
	
	x = convLayer(x, 128, 3)
	x = convLayer(x, 128, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = tf.keras.layers.MaxPool2D(pool_size = 2, padding = 'same')(x)
	skip = tf.pad(skip, [[0, 0], [0, 0], [0, 0], [64, 64]]) # linear projection
	res3 = x
	
	x = convLayer(x, 256, 3, 2)
	x = convLayer(x, 256, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = x
	
	x = convLayer(x, 256, 3)
	x = convLayer(x, 256, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = tf.pad(x, [[0, 0], [0, 0], [0, 0], [128, 128]])
	res4 = x
	
	x = convLayer(x, 512, 3, dilation = 2)
	x = convLayer(x, 512, 3, dilation = 2)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = x
	
	x = convLayer(x, 512, 3, dilation = 2)
	x = convLayer(x, 512, 3, dilation = 2)
	
	x = tf.keras.layers.Add()([x, skip])
	
	x = convLayer(x, 256, 3)
	x = convLayer(x, 256, 3)
	
	x = tf.keras.layers.Add()([x, res4])
	x = tf.keras.layers.UpSampling2D()(x)
	
	x = convLayer(x, 128, 3)
	
	x = tf.keras.layers.Add()([x, res3])
	x = tf.keras.layers.UpSampling2D()(x)
	
	x = convLayer(x, 64, 3)
	
	x = tf.keras.layers.Add()([x, res2])
	x = tf.keras.layers.UpSampling2D()(x)
	
	x = convLayer(x, 32, 3)
	
	outputs = []
	
	if outVectors:
		outputs.append(coordsOutPut(x))
	
	return tf.keras.Model(inputs = xIn, outputs = outputs, name = modelName)

#Legg til enkel model 


def coordsOutPut(x): # add coordinate output layer
	coords = tf.keras.layers.Conv2D(18, (1,1), name = 'coordsOut', kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0), padding = 'same')(x)
	#coords = tf.keras.layers.BatchNormalization(name = 'batchCoords')(coords)
	#coords = tf.keras.layers.Activation('relu')(coords)
	return coords

def convLayer(x, numFilters, kernelSize, strides = 1, dilation = 1):
	x = tf.keras.layers.Conv2D(numFilters, kernelSize, strides = strides, kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0), padding = 'same', dilation_rate = dilation)(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation('relu')(x)
	
	return x

def trainModel(batchSize = 2, optimizer = tf.keras.optimizers.Adam, learning_rate = 0.01, losses = None, metrics = ['accuracy'], modelName = 'stvNet_weights', epochs = 1, loss_weights = None, outVectors = True, outClasses = False, dataSplit = False, altLabels = True, augmentation = True): # train and save model weights
	
	
	#model = modelStruct(outVectors = outVectors, outClasses = outClasses, modelName = modelName)
	model = stvNet()
	model.summary()
	model.compile(optimizer = optimizer(learning_rate = learning_rate), loss ='mean_absolute_error', metrics = metrics, loss_weights = loss_weights)
	
	trainData, validData = None, None
	 
	logger = tf.keras.callbacks.CSVLogger("models\\history\\" + modelName + "_" + "_history.csv", append = True)
	#evalLogger = tf.keras.callbacks.CSVLogger("models\\history\\" + modelName + "_" + modelClass + "_eval_history.csv", append = True)
	
	history, valHistory = [], []
	
	
	dataset = data.coordsTrainingGenerator(batchSize)
	
	hist = model.fit(dataset, steps_per_epoch=1, verbose=1, epochs=2)
		#modelGen(modelClass, batchSize, masterList = trainData, altLabels = altLabels, augmentation = augmentation), steps_per_epoch = math.ceil(len(trainData) / batchSize), max_queue_size = 2, callbacks = [logger])
	history.append(hist.history)

	#Save model 

	return model

