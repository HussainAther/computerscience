import numpy as np
import random

"""
Whale optimization algorithm (WAO) based on how whales hunt for food.

Based off: S. Mirjalili, A. Lewis                                    
           The Whale Optimization Algorithm,                         
           Advances in Engineering Software , in press,              
           DOI: http://dx.doi.org/10.1016/j.advengsoft.2016.01.008  
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
        a=2-t*((2/maxiter)) # Decrease linearly frmo 2 to 0.
        a2 = -1+t*((-1/maxiter)) # a2 linearly decreases from -1 to -2 to calculate t.
        for i in range(positions[0]):
            r1 = random.uniform(0,1) # random numbers in [0,1]
            r2 = random.uniform(0,1)
