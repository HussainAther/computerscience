import numpy as np

from random import seed
from random import random

"""
Neural network backpropagation used for classification are part of the supervised learning
method for multilayer feed-forward artificial neural networks. Using the ways
neural cells process informaiton, we model a given function by modifying internal
weights of inputs signals to give an output signal. We use the error between output and 
a known expected output to modify the internal state.

The seeds dataset involves the prediction of species given measurements seeds from different varieties of wheat.

There are 201 records and 7 numerical input variables. It is a classification problem with 3 output classes. The scale for each numeric input value vary, so some data normalization may be required for use with algorithms that weight inputs like the backpropagation algorithm.
"""

def initialize_network(n_inputs, n_hidden, n_outputs):
    """
    Initialize a neural network with inputs, hidden neurons, and outputs.
    """
    network = list()
    hidden_layer = [{"weights" : [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{"weights" : [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

seed(1234)
network = initialize_network(2, 1, 2)
for layer in network:
    print(layer)


def activate(weights, inputs):
    """
    Calculate neuron activation for weights and input.
    Used in forward propagation.
    """
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

def transfer(activation):
    """
    Transfer the activation to output using the sigmoid
    activation lgoistic function. 
    """
    return 1.0 / (1.0 + np.exp(-activation))
