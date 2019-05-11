from random import choice
from itertools import combinations

"""
In computer science and graph theory, Karger's algorithm is a 
randomized algorithm to compute a minimum cut of a connected graph.
We perform a contraction then a minimum cut.
"""

def contract(graph, u, v):
    """
    During the contraction step, we contract an edge
    to create a new node.
    """
    aux, w = [], f"{u},{v}"
    for x, y in graph:
        x = w if x in [u, v] else x
        y = w if y in [u, v] else y
        if x < y:
            aux.append((x, y))
        elif x > y:
            aux.append((y, x))
    return aux
