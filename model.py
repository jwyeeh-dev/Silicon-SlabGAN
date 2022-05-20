import sys

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D, AvgPool2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

class Discriminator(object):
    def __init__(self, width = 28, height= 28, channels = 1, latent_size=100):
        self.CAPACITY = width*height*channels
        self.SHAPE = (width,height,channels)
        self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)


        self.Discriminator = self.model()
        self.Discriminator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER, metrics=['accuracy'] )
        self.save_model()
        self.summary()

    def model(self):
        model = Sequential()
        # Front side of models
        model.add(Conv2D(filters = 512, kernel_size = (1,3), strides = 1, padding = 0))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(filters = 512, kernel_size = (1,1), strides = 1, padding = 0))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(filters = 256, kernel_size = (1,1), strides = 1, padding = 0))
        model.add(LeakyReLU(alpha = 0.2))
        # Pooling layers
        model.add(AvgPool2D(pool_size= (1000,1)))
        # Backward side of models
        model.add(Dense(500, activation='leakyReLU'))
        model.add(Dense(200, activation='leakyReLU'))
        model.add(Dense(10, activation='leakyReLU'))

        return model

    def summary(self):
        return self.Disciminator.summary()

    def save_model(self):
        plot_model(self.Discriminator.model, to_file='/data/Discriminator_Model.png')