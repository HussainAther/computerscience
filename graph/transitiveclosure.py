
"""
Given a directed graph (V, E) with vertex set V = {1, 2,...n} we might want to determine
whether G contains a path from i to j for all vertex pairs i, j in the set V. Transitive closure
of the graph G is G* = (V, E*) where E* = {(i, j): there is ap ath from vertex i to vertex j in G}
"""

def tc(g):
    """
    Compute the matrix T by the Transitive-Closure procedure on a sample graph g that is a tuple
    of (vertices, edges).
    """
    v = g[0] # vertices array
    e = g[1] # edges array
    n = abs(v)
    s = (n,n)
    T = np.zeros(s) # output matrix T
