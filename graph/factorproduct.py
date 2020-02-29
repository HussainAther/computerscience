"""
Factor-product (factor product)
Use a factor (the basic data structure) or a multi-dimensional table
with an entry for each possible assignment to the variables. For storing multidimensional
tables, we can flatten them into a single array in computer memory. For each variable,
we can store its cardinality and its stride (step size in the factor).
"""

def fp(phi1, phi2):
    """
    For phi1 over scope X1 and phi2 over scope X2 (factors represented
    as a flat array with strides for the variables), perform the 
    factor-product operation.   
    """
