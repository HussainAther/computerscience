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
            self._input_sample = tf.placeholder(tf.float32, [visible_dim], name = "input_sample")

            # Gibbs Sampling
            input_matrix = tf.transpose(tf.stack([self._input_sample for i in range(hidden_dim)]))
            _hidden_probabilities = tf.sigmoid(tf.add(tf.multiply(input_matrix, self._weights), tf.stack([self._hidden_biases[0] for i in range(visible_dim)])))
            self._hidden_states = self.callculate_state(_hidden_probabilities)
            _visible_probabilities = tf.sigmoid(tf.add(tf.multiply(self._hidden_states, self._weights), tf.transpose(tf.stack([self._visible_biases[0] for i in range(hidden_dim)]))))
            self._visible_cdstates = self.callculate_state(_visible_probabilities)
            self._hidden_cdstates = self.callculate_state(tf.sigmoid(tf.multiply(self._visible_cdstates, self._weights) + self._hidden_biases))

            # State matrix
            positive_gradient_matrix = tf.multiply(input_matrix, self._hidden_states)
            negative_gradient_matrix = tf.multiply(self._visible_cdstates, self._hidden_cdstates)
            new_weights = self._weights
            new_weights.assign_add(tf.multiply(positive_gradient_matrix, self._leraning_rate))
            new_weights.assign_sub(tf.multiply(negative_gradient_matrix, self._leraning_rate))

            # Training
            self._training = tf.assign(self._weights, new_weights) 

            # Initilize session and run it
            self._sess = tf.Session()
            initialization = tf.global_variables_initializer()
            self._sess.run(initialization)

    def train(self, v):
        """
        Train on input vectors v.
        """
        for iter_no in range(self._num_iter):
            for i in v:
                self._sess.run(self._training,
                               feed_dict={self._input_sample: i})
