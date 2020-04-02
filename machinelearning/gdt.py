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

with tf.name_scope("Placeholders_examples"):
    # Default placeholder that can be arbitrary float32
    # scalar, vertor, matrix, etc.
    arbitrary_input = tf.placeholder("float32")

    # Input vector of arbitrary length
    input_vector = tf.placeholder("float32", shape=(None,))

    # Input vector that _must_ have 10 elements and integer type
    fixed_vector = tf.placeholder("int32", shape=(10,))

    # Matrix of arbitrary n_rows and 15 columns
    # (e.g. a minibatch of your data table)
    input_matrix = tf.placeholder("float32", shape=(None, 15))
    
    # You can generally use None whenever you don"t need a specific shape
    input1 = tf.placeholder("float64", shape=(None, 100, None))
    input2 = tf.placeholder("int32", shape=(None, None, 3, 224, 224))

    # elementwise multiplication
    double_the_vector = input_vector*2

    # elementwise cosine
    elementwise_cosine = tf.cos(input_vector)

    # difference between squared vector and vector itself plus one
    vector_squares = input_vector**2 - input_vector + 1

my_vector =  tf.placeholder("float32", shape=(None,), name="VECTOR_1")
my_vector2 = tf.placeholder("float32", shape=(None,))
my_transformation = my_vector * my_vector2 / (tf.sin(my_vector) + 1)
print(my_transformation)
dummy = np.arange(5).astype("float32")
print(dummy)
my_transformation.eval({my_vector: dummy, my_vector2: dummy[::-1]})
writer.add_graph(my_transformation.graph)
writer.flush()
