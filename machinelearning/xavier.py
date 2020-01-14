import numpy as np
import tensorflow as tf

"""
Xavier initialization
"""

w = np.random.randn(layer_size[l],layer_size[l-1])*np.sqrt(1/layer_size[l-1])
