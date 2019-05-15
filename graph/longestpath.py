"""
Suppose that we are given a directed acyclic graph G = (V, E)G=(V,E) with 
real-valued edge weights and two distinguished vertices s and t. Describe 
a dynamic-programming approach for finding a longest weighted simple path 
from s to t. What does the subproblem graph look like? What is the efficiency 
of your algorithm?
"""

def longestpath(g, u, t):
    """
    For a graph g, calculate the longest path distance from vertex u to t 
    that returns the tuple (dist, next) in which dist is the memoized array 
    that has the subproblem solutions and next has the path information.
    """
