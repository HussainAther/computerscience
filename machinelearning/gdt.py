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

with tf.name_scope("MSE"):
    y_true = tf.placeholder("float32", shape=(None,), name="y_true")
    y_predicted = tf.placeholder("float32", shape=(None,), name="y_predicted")
    mse = tf.reduce_mean(tf.squared_difference(y_predicted, y_true))

def compute_mse(vector1, vector2):
    """
    Mean square error
    """
    return mse.eval({y_true: vector1, y_predicted: vector2})

writer.add_graph(mse.graph)
writer.flush()

# Rigorous local testing of MSE implementation
import sklearn.metrics
for n in [1, 5, 10, 10**3]:
    elems = [np.arange(n), np.arange(n, 0, -1), np.zeros(n),
             np.ones(n), np.random.random(n), np.random.randint(100, size=n)]
    for el in elems:
        for el_2 in elems:
            true_mse = np.array(sklearn.metrics.mean_squared_error(el, el_2))
            my_mse = compute_mse(el, el_2)
            if not np.allclose(true_mse, my_mse):
                print("mse(%s,%s)" % (el, el_2))
                print("should be: %f, but your function returned %f" % (true_mse, my_mse))
                raise ValueError("Wrong result")

# Creating a shared variable
shared_vector_1 = tf.Variable(initial_value=np.ones(5),
                              name="example_variable")
# Initialize variable(s) with initial values
s.run(tf.global_variables_initializer())

# Evaluating the shared variable
print("Initial value", s.run(shared_vector_1))

# Setting a new value
s.run(shared_vector_1.assign(np.arange(5)))

# Getting that new value
print("New value", s.run(shared_vector_1))
