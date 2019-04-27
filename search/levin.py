import numpy as np

"""
The Universal Levin (levin) search lets us use the idea that we may run two algorithms
at half-speed to determine which one finishes first. We determine computational time
of algorithms by determining the logarith of the running time of a meta-algorithm
that verifies whether the output is a solution for the algorithms that it oversees. 
This Levin complexity is a time-bounded version of Kolmogorov compplexity.
"""

def levin(x):
    """
    For a list of algorithms x in tuple form of (time, length), compute the Levin complexity. 
    """
    summ = 0
    for t, l in x: # for the time and length of each algorithm
        summ += l + np.log(t)
    return summ
