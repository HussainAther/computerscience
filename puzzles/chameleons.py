"""
There are n black, m green, and k brown chameleons on a deserted island. 
When two chameleons of different colors meet they both change their color 
to the third one (e.g., black and green chameleons become brown). For each 
choice of n, m, and k decide whether it is possible that after some time 
all the chameleons on the island are the same color (if you think that it 
is always possible, check the case n = 1, m = 3, and k = 5).
"""

def chameleons(n, m, k):
    """
    For a given input choice of n, m, and k for black, green, and brown chameleons
    respectively, determine whether all the chameleons will reach the same color
    after some times.

    We observe that, if and only if at least two of the color quantities have
    the same remainder after dividing by 3, then we can reduce to a single color. 
    """
    
