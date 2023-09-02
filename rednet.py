from custom_layers import *


class RedNet:
    """
    This class creates a RedNet DNN - a very deep 
    very deep Residual Encoder-Decoder Network.
    Main Attributes:
    input_shape - shape of input to the network
    num_blocks - number of convolutional/deconvolutional blocks (min 2)
    layers_per_block - number of convolutional/deconvolutional per block
    """

    def __init__(self, input_shape, num_blocks, layers_per_block, 
                 filters, kernel_size, strides, padding, activation) -> None:
        
        self.input_shape = input_shape
        self.num_blocks = num_blocks
        self.layers_per_block = layers_per_block
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
    

    def get_rednet(self):
        conv_group = ConvolutionalGroup(filters=self.filters, kernel_size=self.kernel_size,
                                        strides=self.strides, padding=self.padding, 
                                        activation=self.activation)
        
        deconv_group = DeConvolutionalGroup(filters=self.filters, kernel_size=self.kernel_size,
                                            strides=self.strides, padding=self.padding, 
                                            activation=self.activation)
        
        last_deconv_group = DeConvolutionalGroup(filters=self.input_shape[-1], kernel_size=self.kernel_size,
                                                 strides=self.strides, padding=self.padding, 
                                                 activation=self.activation)
        
        skip = SkipConnection()
        
        input_layer = keras.Input(shape=self.input_shape)

        ### Creating convolutional layers ###
        conv_layers_list = []
        conv_layers_list.append(input_layer)

        for i in range(self.num_blocks):
            conv_layers_list.append(conv_group.get_convolution_group(num_convs=self.layers_per_block, 
                                                                     preceding_layer=conv_layers_list[i]))
        
        
        ### Creating deconvolutional layers and skip layers ###

        # First deconvolution group is not succeeded by a skip layer
        deconv_layer = deconv_group.get_de_convolution_group(num_de_convs=self.layers_per_block, 
                                                             preceding_layer=conv_layers_list[-1])
        
        # Second deconvolution group is succeeded by a skip layer
        deconv_layer = deconv_group.get_de_convolution_group(num_de_convs=self.layers_per_block, 
                                                             preceding_layer=deconv_layer)
        skip_layer = skip.get_skip_layer(conv_layer=conv_layers_list[-3], de_conv_layer=deconv_layer)
        skip_layer = tf.keras.layers.ReLU()(skip_layer)

        idx = 3
        while (idx < len(conv_layers_list)):
            idx += 1
            if (idx == len(conv_layers_list)):
                # Last Block
                deconv_layer = last_deconv_group.get_de_convolution_group(num_de_convs=self.layers_per_block,
                                                                          preceding_layer=skip_layer)
                skip_layer = skip.get_skip_layer(conv_layer=conv_layers_list[-idx], de_conv_layer=deconv_layer)
            else:
                deconv_layer = deconv_group.get_de_convolution_group(num_de_convs=self.layers_per_block, 
                                                                    preceding_layer=skip_layer)
                skip_layer = skip.get_skip_layer(conv_layer=conv_layers_list[-idx], de_conv_layer=deconv_layer)
                skip_layer = tf.keras.layers.ReLU()(skip_layer)

        skip_layer = tf.keras.layers.ReLU()(skip_layer)        
        output_layer = tf.keras.layers.Subtract()([skip_layer, input_layer]) # To learn the residual
        
        model = keras.Model(inputs=input_layer, outputs=output_layer, name="RedNet")

        return model
