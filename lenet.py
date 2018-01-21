# -*- coding: utf-8 -*-
"""
Implementing the LeNet ANN Architecture for recognizing handwritten digits
"""
# Load in the required packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class LeNet:
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
        
        # if we are using "channels first". update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            
        # first set of CONV, Relu, POOL layers
        model.add(Conv2D(filters=20, kernel_size=5, padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        
        # second set of CONV, Relu, POOL layers
        model.add(Conv2D(filters=50, kernel_size=5, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=2, strides=2))
            
        # Flatten the input volume using a fully connected layer with 500 nodes
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        
        # softmax classifier, 'classes' is variable that has number of output labels
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        # return the constructed network architecture
        return model
            
        
