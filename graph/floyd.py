import sys

"""
We use the Floyd-Warshall algorithm (Floyd Warshall) to find the shortest path
between all pairs. We find the shortest distances between every pair of vertices
in a given edge weighted directed graph.
"""

inf = sys.maxint # get as high of a value we can

v = 5 # number of vertices in the graph

g = [[0,   5,  inf, 10],
     [inf,  0,  3,  inf],
     [int, inf, 0, 1],
     [inf, inf, inf, 0]] # some graph

def fw(g):
    """
    Input a graph g with the format as given above  
    """
    d = map(lambda i : lambda j: j, i) g) # output result

    for k in range(v): # for each vertex
         for i in range(v):
              for j in range(v):
                   d[i][j] = min(d[i][j], d[i][k] + d[k][j]) # get the minimum of the potential paths
    return d
