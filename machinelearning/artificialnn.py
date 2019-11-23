import numpy as np

from scipy.stats import truncnorm

"""
Artificial neural network (ANN) implementation.
"""

@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)
activation_function = sigmoid
