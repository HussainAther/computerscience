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
