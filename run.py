#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 19:35:29 2017

@author: josh
"""

import numpy as np
import pandas as pd

# Data prep, convert pixels to array from string
data = pd.read_csv("fer2013/fer2013.csv")
image_h, image_w = 48, 48

train = data.loc[data["Usage"] == "Training"]
validate = data.loc[data["Usage"] == "PublicTest"]
test = data.loc[data["Usage"] == "PrivateTest"]

train_X = [np.fromstring(image, dtype=int, sep=" ").reshape((
        image_h, image_w)) for image in train["pixels"]]
train_y = train["emotion"]
validate_X = [np.fromstring(image, dtype=int, sep=" ").reshape((
        image_h, image_w)) for image in validate["pixels"]]
validate_y = validate["emotion"]
test_X = [np.fromstring(image, dtype=int, sep=" ").reshape((
        image_h, image_w)) for image in test["pixels"]]
test_y = test["emotion"]


# map of encoded emotions
emotions = {
    "0": "Angry",
    "1": "Disgust",
    "2": "Fear",
    "3": "Happy",
    "4": "Sad",
    "5": "Surprise",
    "6": "Neutral"
}

# Prepare the images for the inception v3 model
import cv2
from keras.applications.inception_v3 import preprocess_input


for i, im in enumerate(train_X):
    im = im.astype(float)
    im = cv2.resize(im, (299,299))
    p_im = preprocess_input(im)
    train_X[i] = p_im

for i, im in enumerate(validate_X):
    im = im.astype(float)
    im = cv2.resize(im, (299,299))
    p_im = preprocess_input(im)
    validate_X[i] = p_im

for i, im in enumerate(test_X):
    im = im.astype(float)
    im = cv2.resize(im, (299,299))
    p_im = preprocess_input(im)
    test_X[i] = p_im
    
train_X = np.asarray(train_X, dtype=np.float32)
validate_X = np.asarray(validate_X, dtype=np.float32)
test_X = np.asarray(test_X, dtype=np.float32)

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- there are 7 classes
predictions = Dense(7, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit(x=train_X, y=train_y, epochs=50, validation_data=(np.array(validate_X), validate_y))

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit(x=np.array(train_X), y=train_y, epochs=50, validation_data=(np.array(validate_X), validate_y))


model_json = model.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model/model.h5")
print("Saved model to disk")