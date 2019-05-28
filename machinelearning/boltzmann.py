import tensorflow as tf

"""
Restricted Boltzmann machine (RBM).
"""

class RBM(object):
    def __init__(self, visible_dim, hidden_dim, learning_rate, number_of_iterations):
        
        self._graph = tf.Graph()
