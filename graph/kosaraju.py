import numpy as np

"""
Kosaraju's algorithm or Kosaraju-Sharir algorithm is a linear-time algorithm to find the
strongly connected components of a directed graph. It takes advantage of how the tranpose graph
(the same graph with the direction of every edge reversed) has exactly the same
strongly connected components as the original grpah.
"""

def DFSU(graph, v, a):
    """
    Perform depth-fisrt search for a graph graph starting at a node v
    with some starting lsit of visited nodes a.
    """
    a[v]= True # mark the current node as visited

    for i in graph[v]: # recursively check for all vertices adjacent.
        if a[i]==False:
            self.DFS(i,a)

def tranpose(g):
    """
    Return the transpose of a graph g.
    """
    result = [[]]
    for i in g: # recursively get adjacent nodes
        for j in g[i]:
            result[i].append(j)
    return result


def isSC(g, v):
    """
    Return True if the graph g is strongly connected with vertices v. Return False otherwise.
    """

    a = [False]*(v) # mark vertices not visited
    
    DFS(g, 0, a) # perform depth-first search

    if any(i == False for i in a): # see if we have visited all vertices
        return False

    gr = tranpose(g) # create a reversed (tranposed) graph.
    
    a = [False]*(v) # again, mark vertices visited

    gr = DFS(gr, 0,v) # DFS for gr

    if any(i == False for i in v):
        return False

    return True

