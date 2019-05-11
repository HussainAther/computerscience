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

def mincut(graph, n):
    """
    Perform a minimum cut so we can partition the vertices
    of our graph into disjoint sets using n**2 number of attempts
    by creating an n x n array.
    """
    components, cost = ["", ""], float("inf")
    # n^2 attempts
    for i in range(n * n):
        aux = graph
        # remove edges one by one
        while len(set(aux)) > 1:
            aux = contract(aux, *choice(aux))
            # min cut so far
            if len(aux) < cost:
                components, cost = aux[0], len(aux)
    return components, cost


"""
Test out a fully connected graph.
"""

nodes_a = [f"A{i}" for i in range(20)]
graph_a = [(u, v) for u, v in combinations(nodes_a, 2)]

"""
Test out some interconnections.
"""

graph_b = [(choice(nodes_a), choice(nodes_b)) for i in range(10)]

"""
Combine them.
"""

graph = graph_a + graph_b


"""
Test it out with n = 40.
"""

components, cost = mincut(graph, 40)
