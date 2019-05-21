"""
We define the Josephus problem as follows. Suppose that n people form 
a circle and that we are given a positive integer m<=n. Beginning with 
a designated first person, we proceed around the circle, removing every 
mth person. After each person is removed, counting continues around the 
circle that remains. This process continues until we have removed all n 
people. The order in which the people are removed from the circle defines 
the (n-m)-Josephus permutation of the integers 1; 2; : : : ; n. 
For example, the .7; 3/-Josephus permutation is h3; 6; 2; 7; 5; 1; 4i.
"""

def josephus(n, k): 
    """
    Recursive call for removing people.
    """ 
    if n == 1: 
        return 1
    else: 
        return (josephus(n - 1, k) + k-1) % n + 1
