import numpy as np

from sklearn.preprocessing import normalize

"""
IsoRank algorithm computes the alignment (similarity) matrix by a random walk-based
method for non-attributed networks.
"""

def isorank(a1, a2, h, alpha, maxiter, tol):
    """
    For two adjacency matrices a1 and a2 with a prior node similarity matrix h, alpha
    decay factor, and maximum number of iterations maxiter. Output S, an alignment matrix
    that represents to what extent node x in a2 is aligned to node y in a1.
    """
    n1 = a1.size # Get the matrix sizes.
    n2 = a2.size
    d1 = 1/sum(a1, 2) # Sum the matrix with 2 to normalize.
    d2 = 1/sum(a2)
    w1 = d1*a1[:, None] # Multiply the normalized matrices with the original matrices.
    w2 = d1*a1[:, None]
    prev = 0
    s = np.ones(n1, n1)/(n2*n1)
    for i in range(maxiter):
        prev = s
        s = alpha*w2*s*W1+(1-alpha)*h
        delta = normalize(s-prev) 
        if delta < tol:
            break
    return delta 
