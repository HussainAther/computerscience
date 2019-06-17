import numpy as np

"""
3-mode tensor with a map that picks the largest magnitude real eigenvalue and numerical 
integration with the forward Euler method.
"""

def tensorapply(T, x):
    """
    For some array tensor T and vector field x, apply the tensor.
    """
    n = len(x)
    y = np.zeros(n)
    for k in range(n):
        y += T[::k] * x * x[k]
    return y

def tensorcollapse(T, x):
    """
    Collapse the tensor.
    """
