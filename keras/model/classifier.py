import sys
from jupyterlab_server import LabConfig
import numpy as np
import tensorflow as tf

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D, Conv2DTranspose, AvgPool2D, ReLU, BatchNormalization
from keras.layers import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import plot_model

class Classifier(object):
    def __init__(self, width = 28, height= 28, channels = 1, latent_size=100):
        # need a adjustment
        pass

    def model_si(self):
        model_si = Sequential()
        model_si.add(Conv2D(filters= 512, kernel_size= (1,3), strides= 1, padding = 0))
        model_si.add(BatchNormalization(momentum= 0.8))
        model_si.add(LeakyReLU(alpha= 0.2))
        model_si.add(Conv2D(filters= 256, kernel_size= (1,1), strides= 1, padding= 0))
        model_si.add(BatchNormalization(momentum= 0.8))
        model_si.add(LeakyReLU(alpha= 0.2))        
        model_si.add(Conv2D(filters= 256, kernel_size= (1,1), strides= 1, padding= 0))
        model_si.add(BatchNormalization(momentum= 0.8))
        model_si.add(LeakyReLU(alpha= 0.2))        
        model_si.add(Conv2D(filters= 2, kernel_size= (1,1), strides= 1, padding= 0))

        return model_si

    def model_cell(self):
        model_cell = Sequential()
        model_cell.add(Conv2D(filters= 64, kernel_size= (1,3), strides= 1, padding= 0))
        model_cell.add(BatchNormalization(momentum=0.8))
        model_cell.add(LeakyReLU(alpha= 0.3))
        model_cell.add(Conv2D(filters= 64, kernel_size= (1,1), strides = 1, padding= 0))
        model_cell.add(BatchNormalization(momentum=0.8))
        model_cell.add(LeakyReLU(alpha= 0.2))



        return model_cell

    def summary(self):
        return self.Generator.summary()

    def save_model(self):
        plot_model(self.Generator.model, to_file='/data/Classifier_Model.png')