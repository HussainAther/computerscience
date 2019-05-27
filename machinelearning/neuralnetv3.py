import numpy as np

"""
Simplified neural network (NN) of a single neuron neural network.
"""

class NeuralNetwork():
    """
    Class for NN functions and parameters.
    """
    def __init__(self):
        np.random.seed(1)
        self.weight_matrix = 2 * np.random.random((3,1))-1 # 3 x 1 weight matrix
   
    def tanh_der(self, x):
        """
        Derivative of tanh used as activation function.
        """ 
        return 1 - np.tanh(x) ** 2
    
    def forprop(self, inputs):
        """
        Forward propagation.
        """
        return np.tanh(np.dot(inputs, self.weight_matrix))

    def train(self, train_inputs, train_outputs, iter):
        """
        Train the neural network with inputs and outputs and the number of iterations iter.
        """
        for i in range(iter):
            output = self.forprop(train_inputs)
            error = train_outputs - output
            adj = np.dot(train_inputs.T, error * self.tanh_der(output)) # adjustment
            self.weight_matrix += adj 
