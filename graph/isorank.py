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
