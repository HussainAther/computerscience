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

def initNN(n_inputs, n_hidden, n_outputs):
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
network = initNN(2, 1, 2)
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

def forprop(network, row):
    """
    For a network and row inputs, propagate forward.
    """
    inputs = row
    for layer in network:
        new_inputs = []
            for neuron in layer:
                activation = activate(neuron["weights"], inputs)
                neuron["output"] = transfer(activation) 
                new_inputs.append(neuron["output"])
                inputs = new_inputs
    return inputs

def backprop(network, expected):
    """
    Backpropagate error and store in neurons for a given network and
    expected values.
    """
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron["weights"][j] * neuron["delta"])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron["output"])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron["delta"] = errors[j] * transfer_derivative(neuron["output"])

def updateWeights(network, row, l_rate):
    """
    Update network weights with error, row inputs, and 
    learning rate l_rate.
    """
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron["output"] for neuron in network[i - 1]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron["weights"][j] += l_rate * neuron["delta"] * inputs[j]
	    neuron["weights"][-1] += l_rate * neuron["delta"]

def train_network(network, train, l_rate, n_epoch, n_outputs):
    """
    Train a network for a fixed number of epochs.
    """
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forprop(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backprop(network, expected)
            updateWeights(network, row, l_rate)
            print(">epoch=%d, lrate=%.3f, error=%.3f" % (epoch, l_rate, sum_error))
