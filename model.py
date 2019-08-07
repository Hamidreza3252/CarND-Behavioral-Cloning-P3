#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))


# ## Behavorial Cloning  
# 
# ### Review Articles
# - [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/index.html#adam)
# 
# ### Enrichment Readings 
# - [Review: SegNet (Semantic Segmentation)](https://towardsdatascience.com/review-segnet-semantic-segmentation-e66f2e30fb96)
# - [Installing TensorFlow Object Detection API on Windows 10](https://medium.com/@marklabinski/installing-tensorflow-object-detection-api-on-windows-10-7a4eb83e1e7b)
# - [Multi-Sensor Data Fusion (MSDF) for Driverless Cars, An Essential Primer
# ](https://medium.com/@lance.eliot/multi-sensor-data-fusion-msdf-for-driverless-cars-an-essential-primer-a1948bb8b57c)
# - [How to validate your deep learning model with the Diffgram SDK — Tutorial](https://medium.com/diffgram/how-to-validate-your-deep-learning-model-with-the-diffgram-sdk-tutorial-22234a9a35?_hsenc=p2ANqtz-_o0BTtZu_UHjEOD4taLJqxrDs0xDP_xl-Do12O-pIoMFjzmoS945j4gYYqt96YCTANNiUtfOuRCPnutqNDwwtgSCRMhQ&_hsmi=74444548)
# - [How do I design a visual deep learning system in 2019?](https://medium.com/diffgram/how-do-i-design-a-visual-deep-learning-system-in-2019-8597aaa35d03?_hsenc=p2ANqtz-_o0BTtZu_UHjEOD4taLJqxrDs0xDP_xl-Do12O-pIoMFjzmoS945j4gYYqt96YCTANNiUtfOuRCPnutqNDwwtgSCRMhQ&_hsmi=74444548)
# 
# ### Useful Tips
# - [A detailed example of how to use data generators with Keras](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)
# - [Writing Custom Keras Generators](https://towardsdatascience.com/writing-custom-keras-generators-fe815d992c5a)
# 
# ### Image Database
# - [A dataset of images containing...](https://www.kaggle.com/moltean/fruits/downloads/fruits.zip/57)
# 
# ### General Tips
# - It is not necessary to use the left and right images to derive a successful model. Recording recovery driving from the sides of the road is also effective.

# **Center Driving**
# 
# So that the car drives down the center of the road, it's essential to capture center lane driving. Try driving around the track various times while staying as close to the middle of the track as possible even when making turns.
# 
# In the real world, the car would need to stay in a lane rather than driving down the center. But for the purposes of this project, aim for center of the road driving.
# 
# **Strategies for Collecting Data**
# 
# Now that you have driven the simulator and know how to record data, it's time to think about collecting data that will ensure a successful model. There are a few general concepts to think about that we will later discuss in more detail:
# 
# - the car should stay in the center of the road as much as possible
# - if the car veers off to the side, it should recover back to center
# - driving counter-clockwise can help the model generalize
# - flipping the images is a quick way to augment the data
# - collecting data from the second track can also help generalize the model
# - we want to avoid overfitting or underfitting when training the model
# - knowing when to stop collecting more data
# 

# In[2]:


# Load pickled data
import pickle

import pandas as pd
import cv2
import numpy as np
from sklearn import preprocessing
import os
from random import shuffle
import glob
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg
import csv

from keras.layers import Input, InputLayer, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, Lambda, Cropping2D
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import keras
from keras import backend as K
from keras.preprocessing import image
from keras.callbacks import EarlyStopping
from keras.models import load_model


# In[3]:


from importlib import reload

import selfDrivingCarModules
reload(selfDrivingCarModules)
from selfDrivingCarModules import Sdc

import dataProcessingModules
reload(dataProcessingModules)
from dataProcessingModules import DataGenerator4Regression


# In[ ]:





# ### Setting up hyper-parameters for data generation modules  

# In[9]:


data_path = "data/Lake-Track/"
csv_file = data_path + "sim-00/driving_log.csv"
# csv_file = data_path + "driving_log-combined.csv"

data_path = "data/"
csv_file = data_path + "driving_log-combined.csv"

x_partitions = {"train": None, "validation": None}
# y_partitions = {"train": None, "validation": None}

batch_size = 64
image_sizes = (160, 320)

params = {"dims": (*image_sizes, 3), 
          "batch_size": batch_size, 
          "n_channels": 1,
          "augment_data": True,
          "rescale_zero_mean": True,
          "shuffle": True}


# In[ ]:





# ### Creating training and validation data generators  

# In[10]:


# x_partitions["train"], x_partitions["validation"], y_partitions["train"], y_partitions["validation"] = \
#     Sdc.generate_partition_ids(data_path, csv_file, validation_split=0.2, limit=64, image_series_type=Sdc.__CENTER_IMAGES__)

x_partitions["train"], x_partitions["validation"], y_values =     Sdc.generate_partition_ids(data_path, csv_file, validation_split=0.2, limit=0, image_series_type=Sdc.__ALL_IMAGES__, 
                              correction_factor=0.1)

training_generator = DataGenerator4Regression(x_partitions["train"], y_values, **params)
validation_generator = DataGenerator4Regression(x_partitions["validation"], y_values, **params)

# testing data generators 
x_data = training_generator[0][0]
y_data = training_generator[0][1]

test_index = 10

print("batch size={0:d} , number of batches={1:d}".format(batch_size, len(training_generator)))

# for augmented, they should be opposite values
print("sample training y_data: {0:0.3f}, {1:0.3f}".format(y_data[test_index], y_data[test_index + batch_size]))

y_data = validation_generator[0][1]
print("sample validation y_data: {0:0.3f}, {1:0.3f}".format(y_data[test_index], y_data[test_index + batch_size]))

# a check-point whether the values are re-scaled or not
print(np.min(x_data[test_index]), np.max(x_data[test_index]))


# In[ ]:





# ### Training New CNN Model from Scratch (No Transfer Learning)  

# In[ ]:





# In[9]:


model = Sdc.generate_model("cnn-01", image_sizes, rescale_input_zero_mean=False)
model.summary()


# In[4]:


import pickle


# In[2]:


model_name = "001-model-conv-6-fc-4-all-aug-crop"
model_filename = "saved-models/" + model_name + ".h5"
history_filename = "saved-models/" + model_name + ".p"

checkpoint_file = model_filename


# In[ ]:


checkpoint = ModelCheckpoint(filepath=checkpoint_file, monitor="val_loss", save_best_only=True)
stopper = EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=5)

model.compile(loss="mse", optimizer="adam")
history = model.fit_generator(generator=training_generator, validation_data=validation_generator, 
                              use_multiprocessing=True, workers=2, epochs=50, callbacks=[checkpoint, stopper])

model.save(model_filename)


# In[5]:


with open(history_filename, "wb") as file_pi:
    pickle.dump(history.history, file_pi)


# In[ ]:





# ### Transfer Learning Model (InceptionV3)  
# 

# In[11]:


# Load our images first, and we'll check what we have
# from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

from keras.layers import Input, Lambda
import tensorflow as tf
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping


# In[ ]:





# In[12]:


resized_input_shape = (139, 139)

freeze_flag = False # `True` to freeze layers, `False` for full training 
weights_flag = "imagenet" # 'imagenet' or None 
preprocess_flag = True # Should be true for ImageNet pre-trained typically 


# Using smaller than the default 299x299x3 input for InceptionV3
# which will speed up training. Keras v2.0.9 supports down to 139x139x3

# input_size = 139

# Using Inception with ImageNet pre-trained weights
inception = InceptionV3(weights=weights_flag, include_top=False, input_shape=(*resized_input_shape, 3))

if (freeze_flag == True):
    for layer in inception.layers:
        layer.trainable = False

# inception.summary()


# In[ ]:





# In[13]:


# Makes the input placeholder layer with image shape
input_ph = Input(shape=(*image_sizes, 3))

preprocessed_input = Cropping2D(cropping=((50,20), (0,0)), input_shape=(*image_sizes, 3))(input_ph)
preprocessed_input = Lambda(lambda image: tf.image.resize_images(     image, (139, 139), method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False))(preprocessed_input)

# preprocessed_input = Lambda(lambda x: x / 255.0 - 0.5, input_shape=(*input_size, 3))(preprocessed_input)

inception_output = inception(preprocessed_input)

# layer_output = Flatten()(inception_output)
layer_output = GlobalAveragePooling2D()(inception_output)

layer_output = Dense(128, activation=None, name="fc1")(layer_output)
layer_output = Dropout(rate=0.20)(layer_output)

layer_output = Dense(64, activation=None, name="fc2")(layer_output)
layer_output = Dropout(rate=0.20)(layer_output)

layer_output = Dense(32, activation=None, name="fc3")(layer_output)
layer_output = Dropout(rate=0.20)(layer_output)

predictions = Dense(1, activation=None, name="fc4")(layer_output)


# In[14]:


model = Model(inputs=input_ph, outputs=predictions, name="cnn-20")
model.compile(optimizer="Adam", loss="mse", metrics=["mse"])
model.summary()


# In[15]:


model_name = "051-model-inception-partial-data-04"
model_filename = "saved-models/" + model_name + ".h5"
history_filename = "saved-models/" + model_name + ".p"

checkpoint_file = model_filename


# In[16]:


checkpoint = ModelCheckpoint(filepath=checkpoint_file, monitor="val_loss", save_best_only=True)
stopper = EarlyStopping(monitor="val_loss", min_delta=1e-6, patience=5)


# In[ ]:


history = model.fit_generator(generator=training_generator, validation_data=validation_generator, 
                              use_multiprocessing=True, workers=2, epochs=100, callbacks=[checkpoint])

model.save(model_filename)

with open(history_filename, "wb") as file_pi:
    pickle.dump(file_pi)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




