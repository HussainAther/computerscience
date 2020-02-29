import numpy as np

"""
Factor-product (factor product)
Use a factor (the basic data structure) or a multi-dimensional table
with an entry for each possible assignment to the variables. For storing multidimensional
tables, we can flatten them into a single array in computer memory. For each variable,
we can store its cardinality and its stride (step size in the factor).
"""

def fp(phi1, X1, phi2, X2):
    """
    For phi1 over scope X1 and phi2 over scope X2 (factors represented
    as a flat array with strides for the variables), perform the 
    factor-product operation.   
    """
    j = 0
    k = 0
    l = []
    psi = []
    for i in range(X1.union(X2)[-1]):
        l[i] = 0 
    for i in range(X1.union(X2)[-1]-1):
        psi[i] = phi1[j]*phi2[k]
        for l in range(X1.union(X2)[-1]):
            l = l + 1
