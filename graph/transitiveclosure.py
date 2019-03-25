"""
Given a directed graph (V, E) with vertex set V = {1, 2,...n} we might want to determine
whether G contains a path from i to j for all vertex pairs i, j in the set V. Transitive closure
of the graph G is G* = (V, E*) where E* = {(i, j): there is a path from vertex i to vertex j in G}
"""

def tc(g):
    """
    Compute the matrix T by the Transitive-Closure procedure on a sample graph g that is a tuple
    of two dimensions.
    """
    r = [i[:] for i in g]
    for k in range(v):
        for i in range(v):  # Pick all vertices as source one by one
            for j in range(v):
                r[i][j] = r[i][j] or (r[i][k] and r[k][j])
    return r


g = [[1, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1]]
