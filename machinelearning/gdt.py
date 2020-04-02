import numpy as np
import tensorflow as tf

"""
Gradient descent trajectory (gdt) in tensorflow 
"""

def sum_python(N):
    """
    Vector sum
    """
    return np.sum(np.arange(N)**2)

# An integer parameter
N = tf.placeholder("int64", name="input_to_your_function")

# A recipe on how to produce the same result
result = tf.reduce_sum(tf.range(N)**2)
