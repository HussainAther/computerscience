import download_utils
import keras, keras.layers as L, keras.backend as K
import keras_utils
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from lfw_dataset import load_lfw_dataset
from sklearn.model_selection import train_test_split

"""
Denoising autoencoders 
"""
