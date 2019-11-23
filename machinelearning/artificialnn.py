import numpy as np

from scipy.stats import truncnorm

"""
Artificial neural network (ANN) implementation.
"""

@np.vectorize
def sigmoid(x):
    """
    Sigmoid function for activation in optimization problems
    """
    return 1 / (1 + np.e ** -x)
activation_function = sigmoid

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    """
    Truncate it with mean, standard deviation, lower and upper bound.
    """
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
