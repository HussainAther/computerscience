import sys

"""
We use Prim's algorithm to consider
the edges that connect two sets and pick the minimum weight
edge from these edges. After picking the edge, move the other
end-point of hte edge to the set containing the minimum spanning tree (MST). 
"""

def mindist(g, k, mst):
    """
    Find the vertex with minimum distance from the set of vertices 
    not in the shortest path tree.
    """
    mini = sys.maxint # initialize some minimum using maxint so we can reduce
    for i in range(g[1]): # for each vertex
         if k[i] < mini and mst[i] == False:
               min_ind = i # get the minimum index
    return min_ind 

def prim(g):
    """
    Return MST using adjacency matrix for graph (tuple of edges and vertices) g.
    """
    v = g[1] # vertices
    k = [sys.maxint] * v # initialize some keys using maxint so we can reduce
    p = [None] * v
    k[0] = 0
    mst = [False] * v
    p[0] = -1
    for i in range(v):
          u = mindist(g, k, mst) # get the minimum distance vertex
          mst[u] = True # set the value true for it
          for j in range(v):
                if g[u][j] > 0 and mst[j] == False:
                    k[v] = g[u][v]
                    p[v] = u
    return p
