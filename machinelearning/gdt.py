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
