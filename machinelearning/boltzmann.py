import tensorflow as tf

"""
Restricted Boltzmann machine (RBM).
"""

class RBM(object):
    def __init__(self, visible_dim, hidden_dim, learning_rate, number_of_iterations):
        
        self._graph = tf.Graph()
        
        # Initialize graph
        with self._graph.as_default():
            self._num_iter = number_of_iterations
            self._visible_biases = tf.Variable(tf.random_uniform([1, visible_dim], 0, 1, name = "visible_biases"))
            self._hidden_biases = tf.Variable(tf.random_uniform([1, hidden_dim], 0, 1, name = "hidden_biases"))
            self._hidden_states = tf.Variable(tf.zeros([1, hidden_dim], tf.float32, name = "hidden_biases"))
            self._visible_cdstates = tf.Variable(tf.zeros([1, visible_dim], tf.float32, name = "visible_biases"))
            self._hidden_cdstates = tf.Variable(tf.zeros([1, hidden_dim], tf.float32, name = "hidden_biases"))
            self._weights = tf.Variable(tf.random_normal([visible_dim, hidden_dim], 0.01), name="weights")
            self._leraning_rate =  tf.Variable(tf.fill([visible_dim, hidden_dim], learning_rate), name = "learning_rate")
