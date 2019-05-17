import numpy as np

from random import seed
from random import random
from csv import reader

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

def train(network, train, l_rate, n_epoch, n_outputs):
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

def loadcsv(filename):
    """
    Load a given csv file.
    """
    dataset = list()
    with open(filename, "r") as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def strColumnToFloat(dataset, column):
    """
    Convert string column to float.
    """
    for row in dataset:
        row[column] = float(row[column].strip())

def strColumnToInt(dataset, column):
    """
    Convert string column to integer.
    """
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

def datasetMinMax(dataset):
    """
    Find the minimum and maximum values for each column.
    """
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

def normalizeDataset(dataset, minmax):
    """
    Rescale the dataset columns to the range 0-1 using the minmax minimum
    and maximum.
    """
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def cvSplit(dataset, n_folds):
    """
    Split a dataset into k folds for cross-validation cross validation.
    """
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def accuracy(actual, predicted):
    """
    Calculate accuracy.
    """
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

seed(1234)
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initNN(n_inputs, 2, n_outputs)
train(network, dataset, 0.5, 20, n_outputs)
