import numpy as np

from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import add
from keras.utils import plot_model

"""
Convolutional neural networks are an important class of learnable representations applicable, 
among others, to numerous computer vision problems. Deep CNNs, in particular, are composed of 
several layers of processing, each involving linear as well as non-linear operators, that are 
learned jointly, in an end-to-end manner, to solve a particular tasks. These methods are now 
the dominant approach for feature extraction from audiovisual and textual data.

This practical explores the basics of learning (deep) CNNs. The first part introduces typical 
CNN building blocks, such as ReLU units and linear filters, with a particular emphasis on 
understanding back-propagation. The second part looks at learning two basic CNNs. The first 
one is a simple non-linear filter capturing particular image structures, while the second one 
is a network that recognises typewritten characters (using a variety of different fonts). These 
examples illustrate the use of stochastic gradient descent with momentum, the definition of an 
objective function, the construction of mini-batches of data, and data jittering. The last part 
shows how powerful CNN models can be downloaded off-the-shelf and used directly in applications, 
bypassing the expensive training process.

Specifically, models that have achieved state-of-the-art results for tasks like image classification 
use discrete architecture elements repeated multiple times, such as the VGG block in the VGG models, 
the inception module in the GoogLeNet, and the residual module in the ResNet.
"""

def vggblock(layer_in, n_filters, n_conv):
    """
    Create VGG bloc for input layer layer_in, number of filters n_filters, and number of 
    convolutions n_conv. 
    """
    for _ in range(n_conv):
        layer_in = Conv2D(n_filters, (3,3), padding="same", activation="relu")(layer_in)
    layer_in = MaxPooling2D((2,2), strides=(2,2))(layer_in) # max pooling layer
    return layer_in

def inceptionmodule(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
    """
    Create a projected inception module for type of input layer layer_in and numbers
    that dictate the input and output of each convolution.
    """
    conv1 = Conv2D(f1, (1,1), padding="same", activation="relu")(layer_in)
    conv3 = Conv2D(f2_in, (1,1), padding="same", activation="relu")(layer_in)
    conv3 = Conv2D(f2_out, (3,3), padding="same", activation="relu")(conv3)
    conv5 = Conv2D(f3_in, (1,1), padding="same", activation="relu")(layer_in)
    conv5 = Conv2D(f3_out, (5,5), padding="same", activation="relu")(conv5)
    pool = MaxPooling2D((3,3), strides=(1,1), padding="same")(layer_in)
    pool = Conv2D(f4_out, (1,1), padding="same", activation="relu")(pool)
    layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out

def residualmodule(layer_in, n_filters):
    """
    For a type of input layer layer_in and number of filters, create an identity
    or projection residual.
    """
    merge_input = layer_in
    # check if the number of filters needs to be increase, assumes channels last format
    if layer_in.shape[-1] != n_filters:
        merge_input = Conv2D(n_filters, (1,1), padding="same", activation="relu", kernel_initializer="he_normal")(layer_in)
    conv1 = Conv2D(n_filters, (3,3), padding="same", activation="relu", kernel_initializer="he_normal")(layer_in)
    conv2 = Conv2D(n_filters, (3,3), padding="same", activation="linear", kernel_initializer="he_normal")(conv1)
    layer_out = add([conv2, merge_input])
    layer_out = Activation("relu")(layer_out)
    return layer_out

# VGG architecture
visible = Input(shape=(256, 256, 3))
layer = vggblock(visible, 64, 2)
layer = vggblock(layer, 128, 2)
layer = vggblock(layer, 256, 4)
model = Model(inputs=visible, outputs=layer)
model.summary()
plot_model(model, show_shapes=True, to_file="multiple_vgg_blocks.png")

# Inception module
layer = inceptionmodule(visible, 64, 96, 128, 16, 32, 32)
layer = inceptionmodule(layer, 128, 128, 192, 32, 96, 64)
model = Model(inputs=visible, outputs=layer)
model.summary()
plot_model(model, show_shapes=True, to_file="inception_module.png")

# Residual model
layer = residualmodule(visible, 64)
model = Model(inputs=visible, outputs=layer)
model.summary()
plot_model(model, show_shapes=True, to_file="residual_module.png")
