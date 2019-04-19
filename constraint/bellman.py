import numpy as np

"""
We use the Bellman equation (bellman) as a necessry condition for optimality
associated with the mathematical optimization method known as dynamic
programming. Richard Bellman defined the principle of optimality as 

"Principle of Optimality: An optimal policy has the property that whatever the initial state and initial decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision."

We let a state at time t by xt. For a decision beginning at time t=0, we take the initial state x0. 
"""

def infhordec(x):
    """
    We create an infinite-horizon decision problem that we can compute the optimal value by maximizing
    this objective function with assumed constraints. 
    """ 
