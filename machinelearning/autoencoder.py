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

# Load data.
download_utils.link_week_4_resources()
X, attr = load_lfw_dataset(use_raw=True, dimx=32, dimy=32)
IMG_SHAPE = X.shape[1:]

# Center images.
X = X.astype('float32') / 255.0 - 0.5

# Split.
X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)
