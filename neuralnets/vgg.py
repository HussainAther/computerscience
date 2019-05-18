import numpy as np

from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
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

def inception_module(layer_in, f1, f2, f3):
    """
    Create naive inception block.
    """
    conv1 = Conv2D(f1, (1,1), padding="same", activation="relu")(layer_in) # 1x1 
    conv3 = Conv2D(f2, (3,3), padding="same", activation="relu")(layer_in) # 3x3
    conv5 = Conv2D(f3, (5,5), padding="same", activation="relu")(layer_in) # 5x5
    pool = MaxPooling2D((3,3), strides=(1,1), padding="same")(layer_in) # Max pooling
    layer_out = np.concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out

visible = Input(shape=(256, 256, 3)) # define model input
layer = vggblock(visible, 64, 2) # add vgg module
layer = vggblock(layer, 128, 2) 
layer = vggblock(layer, 256, 4)
model = Model(inputs=visible, outputs=layer) # create mode
model.summary() # summarize model
plot_model(model, show_shapes=True, to_file="multiplevggblocks.png") # plot model architecture
