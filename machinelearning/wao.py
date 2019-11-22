import numpy as np
import random

"""
Whale optimization algorithm (WAO) based on how whales hunt for food.
"""

def wao(sano, maxiter, lb, ub, dim, fobj)
    """
    The optimization algorithm with sano as number of search agents,
    lower boundary lb, upper boundary ub (for the boundary of the search space),
    dim dimensions and fobj objective function.
   
    Return the lscore leader score, lpos leader position, and conv convergence curve.
    """
    lpos = np.zeros(dim)
    lscore = np.inf
    positions = [sano, dim, ub, lb]
    conv = np.zeros(maxiter)
    t=0
    while t < maxiter:
        for i in range(positions[0]):
            flag4ub = positions[i][:] > ub
            flag4lb = posiitons[i][:] < lb
            positions[i][:] = (positions[i][:]*(-(flag4ub+flag4lb))) + ub*flag4ub+lb*flag4lb
