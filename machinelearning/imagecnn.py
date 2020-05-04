import grading
import grading_utils
import json
import keras
import keras, keras.layers as L
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import re
import tensorflow as tf
import time
import tqdm
import utils
import zipfile

from collections import defaultdict
from random import choice

# Image captioning convolutional neural network (cnn)

IMG_SIZE = 299

def get_cnn_encoder():
    """
    Encoding architecture
    """
    K.set_learning_phase(0)
    model = keras.applications.InceptionV3(include_top=False)
    preprocess_for_model = keras.applications.inception_v3.preprocess_input
    model = keras.engine.training.Model(model.inputs, keras.layers.GlobalAveragePooling2D()(model.output))
    return model, preprocess_for_model
