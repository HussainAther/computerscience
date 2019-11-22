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
            # Return the search agents beyond the boundaries. 
            flag4ub = positions[0][i] > ub
            flag4lb = posiitons[0][i] < lb
            positions[0][i] = (positions[0][i]*(-(flag4ub+flag4lb))) + ub*flag4ub+lb*flag4lb
            # Calculate the objective function for each search agent.
            fitness = fobj(positions[0][i])
            # Update the leadaer.
            if fitness < lscore: 
                lscore = fitness
                lpos = positions[0][i]
