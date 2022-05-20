import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D, Conv2DTranspose, AvgPool2D, ReLU, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

class Generator(object):
    def __init__(self, width = 28, height= 28, channels = 1, latent_size=100):
        # need a adjustment
        self.W = width
        self.H = height
        self.C = channels
        self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)

        self.LATENT_SPACE_SIZE = latent_size
        self.latent_space = np.random.normal(0,1,(self.LATENT_SPACE_SIZE,))

        self.Generator = self.model()
        self.Generator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)
        self.save_model()
        self.summary()

    def model(self, block_starting_size=256,num_blocks=3):
        model = Sequential()
        
        block_size = block_starting_size 
        
        model.add(Dense(block_size, input_shape=(self.LATENT_SPACE_SIZE,)))
        model.add(ReLU(alpha=0.2))

        for i in range(num_blocks-1):
            model.add(Conv2DTranspose(filters = 256, kernel_size = (1,3), strides = 1, padding = 0))        
            model.add(BatchNormalization(momentum=0.8))
            model.add(ReLU(alpha=0.2))
            block_size = block_size * 2

        model.add(Dense(self.W * self.H * self.C, activation='sigmoid'))
        model.add(Reshape((self.W, self.H, self.C)))
        
        return model

    def summary(self):
        return self.Generator.summary()

    def save_model(self):
        plot_model(self.Generator.model, to_file='/data/Generator_Model.png')