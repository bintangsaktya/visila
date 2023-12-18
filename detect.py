import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import matplotlib.pyplot as plt
import scipy
from scipy import linalg  # noqa: F401
from scipy import ndimage  # noqa: F401
import cv2
import numpy as np

# Deep Learning CNN model to recognize face
'''This script uses a database of images and creates CNN model on top of it to test
   if the given image is recognized correctly or not'''

'''####### IMAGE PRE-PROCESSING for TRAINING and TESTING data #######'''

# Specifying the folder where images are present
TrainingImagePath= r"F:\KECERDASAN BUATAN\tensorfloww\skin-disease-datasaet\train_set"
TestImagePath = r"F:\KECERDASAN BUATAN\tensorfloww\skin-disease-datasaet\test_set"

from keras.preprocessing.image import ImageDataGenerator
# Understand more about ImageDataGenerator at below link
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
 
# Defining pre-processing transformations on raw images of training data
# These hyper parameters helps to generate slightly twisted versions
# of the original image, which leads to a better model, since it learns
# on the good and bad mix of images
train_datagen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=10,
        width_shift_range=0.05,
                             height_shift_range=0.05,
                                     shear_range=0.05,
                                     zoom_range=0.05,
                                     horizontal_flip=True,
                                     fill_mode='nearest')
 
# Defining pre-processing transformations on raw images of testing data
# No transformations are done on the testing images
test_datagen = ImageDataGenerator(rescale=1/255)
 
# Generating the Training Data
training_set = train_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')


# Generating the Testing Data
test_set = test_datagen.flow_from_directory(
        TestImagePath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')
 
# Printing class labels for each face
test_set.class_indices

## BIKIN MODELNYA ##
'''############ Creating lookup table for all faces ############'''
# class_indices have the numeric tag for each face
TrainClasses=training_set.class_indices

# Storing the face and the numeric tag for future reference
ResultMap={}
for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
    ResultMap[faceValue]=faceName

# Saving the face map for future reference
import pickle
with open("ResultsMap.pkl", 'wb') as fileWriteStream:
    pickle.dump(ResultMap, fileWriteStream)

# The model will give answer as a numeric tag
# This mapping will help to get the corresponding face name for it
print("Mapping of Face and its ID",ResultMap)

# The number of neurons for the output layer is equal to the number of faces
OutputNeurons=len(ResultMap)
print('\n The Number of output neurons: ', OutputNeurons)

### MODELL ###
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
# GRADED FUNCTION: create_model
def create_model():
  # DEFINE A KERAS MODEL TO CLASSIFY CATS V DOGS
  # USE AT LEAST 3 CONVOLUTION LAYERS

  ### START CODE HERE

  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64,64,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(OutputNeurons, activation='softmax')
  ])


  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  ### END CODE HERE
  model.summary()
  return model
