import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras


### This file contains the following classes that instantiate the respective group of layers
### using the get function:
### (a) 2D Convolutional layers
### (b) 2D Deconvolutional layers
### (c) 2D Skip layers (connections)


class ConvolutionalGroup:
    """
    This class returns last layer of a group of 2D-convolutional layers
    created using the get function. Returns atleast 1 layer.
    Main Atrribute:
    num_convs --- the number of convolutional layers needed in the group
    preceding_layer --- layer preceding the group
    """

    def __init__(self, filters, kernel_size, strides, padding, activation) -> None:
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
    

    def get_convolution_group(self, num_convs, preceding_layer):
        conv_layer = tf.keras.layers.Conv2D(filters = self.filters, kernel_size = self.kernel_size,
                                            strides = self.strides, padding = self.padding,
                                            activation = self.activation)(preceding_layer)

        for _ in range(num_convs - 1):
            conv_layer = tf.keras.layers.Conv2D(filters = self.filters, kernel_size = self.kernel_size,
                                                strides = self.strides, padding = self.padding,
                                                activation = self.activation)(conv_layer)

        return conv_layer



class DeConvolutionalGroup:
    """
    This class returns the last layer of a group of 2D de-convolutional layers
    created using the get function. Returns atleast 1 layer.
    Main Atrribute:
    num_de_convs --- the number of de-convolutional layers needed in the group
    preceding_layer --- layer preceding the group
    """

    def __init__(self, filters, kernel_size, strides, padding, activation) -> None:
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
    

    def get_de_convolution_group(self, num_de_convs, preceding_layer):
        de_conv_layer = tf.keras.layers.Conv2DTranspose(filters = self.filters, kernel_size = self.kernel_size,
                                                        strides = self.strides, padding = self.padding,
                                                        activation = self.activation)(preceding_layer)
        
        for _ in range(num_de_convs - 1):
            de_conv_layer = tf.keras.layers.Conv2DTranspose(filters = self.filters, kernel_size = self.kernel_size,
                                                            strides = self.strides, padding = self.padding,
                                                            activation = self.activation)(de_conv_layer)

        return de_conv_layer



class SkipConnection:
    """
    This class instantiates a skip layer through the get function.
    This skip layer adds a skip connection between the
    input convolutional and deconvolutional layers
    """

    def __init__(self) -> None:
        pass

    def get_skip_layer(self, conv_layer, de_conv_layer):
        return tf.keras.layers.Add()([conv_layer, de_conv_layer])
