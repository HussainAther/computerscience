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

            # State matrix for contrast divergence
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

    def calculate_state(self, probability):
        """
        Calculate state using probability.
        """
        return tf.floor(probability + tf.random_uniform(tf.shape(probability), 0, 1))

"""
Deep Boltzmann machine (dbm) using scikit-learn.
"""

import argparse
import time
import cv2

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

def nudge(X, y):
    """
    Initialize the translations to shift the image X one pixel up, down,
    left, and right. then initialize the new data matrix and targets y.
    """
    translations = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    data = []
    target = []
    for (image, label) in zip(X, y):
        """
        Reshape the image from a feature vector of 784 raw
        pixel intensities to a 28x28 image
        """
        image = image.reshape(28, 28)
	for (tX, tY) in translations:
	    # translate the image
	    M = np.float32([[1, 0, tX], [0, 1, tY]])
	    trans = cv2.warpAffine(image, M, (28, 28))
	    # update the list of data and target
	    data.append(trans.flatten())
	    target.append(label)
	return (np.array(data), np.array(target)) # return a tuple of the data matrix and targets

# Initialize RBM and logistic regression
rbm = BernoulliRBM()
logistic = LogisticRegression()
classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])

params = {
	"rbm__learning_rate": [0.1, 0.01, 0.001],
	"rbm__n_iter": [20, 40, 80],
	"rbm__n_components": [50, 100, 200],
	"logistic__C": [1.0, 10.0, 100.0]}

# Grid search
params = {"C": [1.0, 10.0, 100.0]}
start = time.time()
gs = GridSearchCV(LogisticRegression(), params, n_jobs = -1, verbose = 1)
gs.fit(trainX, trainY)

# Get best model
bestParams = gs.best_estimator_.get_params()

