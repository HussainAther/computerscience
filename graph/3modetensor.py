import numpy as np

from sklearn.preprocessing import normalize

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
    n = len(x)
    y = np.zeros([n, n])
    for k in range(n):
        y += T[::k] * x[k]
    return y

def dxdt(u):
    """
    Find the derivative of some function u.
    """
    f = np.linalg.eigen(tensorcollapse(T, u))
    ind = [abs(np.real(i)) for i in permutations(F)]
    v = F[:ind]
    return np.sign(v[0])

def foreul(T, h, niter):
    """
    Forward Euler method to pick the largest real eigenvalue and numerical integration. 
    """
    x =  
