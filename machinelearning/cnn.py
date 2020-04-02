import keras
import keras_utils
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.datasets import cifar10

"""
Convolutional neural network (CNN) in CIFAR-10 dataset
"""

# Load data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

NUM_CLASSES = 10
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", 
                   "dog", "frog", "horse", "ship", "truck"]

# Normalize inputs.
x_train2 = x_train / 255. - 0.5
x_test2 = x_test / 255. - 0.5

# Convert class labels to one-hot encoded, should have shape (?, NUM_CLASSES).
y_train2 = keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
y_test2 = keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)
