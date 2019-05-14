import numpy as np

"""
You are given an 8 × 8 table of natural numbers. In any one step, you can 
either double each of the numbers in any one row, or subtract 1 from each 
of the numbers in any one column. Devise an algorithm that transforms the 
original table into a table of all zeros. What is the running time of your 
algorithm?
"""

def solve(m):
    """
    For some matrix m, transform it into all zeros.
    """
    lowestone = np.inf
    for c in m: # for each column
        for i, j in enumerate(m[c]): # get the lowest non-zero non-2 item in the first column
            if j != 2 and j!= 0 and j < lowest one:
                lowestone = j
        for i in range(lowestone-1): # subtract this column lowest-1 times
            m[c] -= 1
    for c in m:
        for j in m:
            if m[c][j] == 1:
                m[c] *= 2
                m[c] -=1 
                m[i] -=1 
