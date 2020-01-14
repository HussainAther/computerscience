import numpy as np
import tensorflow as tf

"""
Xavier initialization
"""

w = np.random.randn(layer_size[l],layer_size[l-1])*np.sqrt(1/layer_size[l-1])

tf.contrib.layers.xavier_initializer(
    uniform=True,
    seed=None,
    dtype=tf.float32
)
tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN", uniform=False, seed=None, dtype=tf.float32)
