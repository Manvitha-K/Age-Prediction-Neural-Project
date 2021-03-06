from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras import layers
from keras.layers import Dropout, Flatten, Dense, Input
from keras import applications
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import RMSprop, SGD, Adam
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import initializers


def baseModel():
	model = Sequential()
	model.add(layers.Conv2D(96, (7, 7), activation='relu', input_shape=(227, 227, 3), kernel_initializer=initializers.random_normal(stddev=0.01)))
	model.add(layers.MaxPooling2D((3, 3), strides=(2,2)))
	model.add(BatchNormalization())

	model.add(layers.Conv2D(256, (5, 5), activation='relu', kernel_initializer=initializers.random_normal(stddev=0.01)))
	model.add(layers.MaxPooling2D((3, 3), strides=(2,2) ))
	model.add(BatchNormalization())

	model.add(layers.Conv2D(384, (3, 3), activation='relu', kernel_initializer=initializers.random_normal(stddev=0.01)))
	model.add(layers.MaxPooling2D((3, 3)))
	
	model.add(layers.Flatten())
	model.add(layers.Dense(512, activation='relu', kernel_initializer=initializers.random_normal(stddev=0.01)))
	model.add(Dropout(0.5))

	model.add(layers.Dense(512, activation='relu', kernel_initializer=initializers.random_normal(stddev=0.01)))
	model.add(Dropout(0.5))
	model.add(layers.Dense(8, activation='softmax'))
	return model




def baseModelDataGen(trainData, validData, testData):
	batchSize = 50
	train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')            	
	train_generator = train_datagen.flow_from_directory(trainData, target_size = (227, 227), batch_size = batchSize, class_mode = "categorical", shuffle = True)

	valid_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest') 
	valid_generator = valid_datagen.flow_from_directory(validData, target_size = (227, 227), batch_size = batchSize, class_mode = "categorical", shuffle = True)
	
	test_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
	test_generator = test_datagen.flow_from_directory(testData,target_size = (227, 227), batch_size = batchSize, class_mode = "categorical", shuffle = True)
	
	return (train_generator, valid_generator, test_generator)
	




def baseTrain(trainData, validData, testData):
        cwd = os.getcwd()
        trainLen = sum(len(files) for _, _, files in os.walk(trainData))
        validLen = sum(len(files) for _, _, files in os.walk(validData))
        testLen =  sum(len(files) for _, _, files in os.walk(testData))
        batch_size = 50
        myModel = baseModel()
        train_generator, valid_generator, test_generator = baseModelDataGen(trainData, validData, testData)
        myModel.compile(optimizer=SGD(lr=0.001, decay = 1e-5, momentum=0.4, nesterov=True),loss='categorical_crossentropy',metrics=['accuracy'])
        callbacks = [ModelCheckpoint(filepath=cwd + '/hdf' + '/age8_base.hdf5', verbose=0, save_best_only=True, save_weights_only=True)]
        history = myModel.fit_generator(train_generator,
                    steps_per_epoch = int(np.ceil(trainLen/batch_size)),
                    epochs=40,
                    validation_data=valid_generator,
                    validation_steps= int(np.ceil(validLen/batch_size)),
                    verbose=1)
        myModel.save(cwd + '/hdf' +'/age8_40_base.h5')
        testAcc = myModel.evaluate_generator(test_generator, int(np.ceil(testLen/batch_size)))
        print(testAcc[0], testAcc[1])
	

if __name__ == '__main__':
	cwd = os.getcwd()
	trainData = cwd + '/data/age_train'
	validData = cwd + '/data/age_valid'
	testData = cwd + '/data/age_test'
	baseTrain(trainData, validData, testData)
	
	
	
	
