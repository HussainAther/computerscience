import itertools

"""
A vertex-cover (vertex cover vertexcover) of an undirected graph (G=(V, E)) is a subset
V' of V such that, if edge (u, v) is an edge of G, then u is in V' or v is in V' or both.
The set V' is said to "cover" the edges of G. A minimum vertex cover is a vertex cover
of the smallest possible size. If every vertex has a cost, we want to minimize the total
cost while covering every edge of the graph.  
"""

class vertexcover:
    """
    For a graph, we want to check vertex covers.
    """
    def __init__(self, graph):
        """
        Initialize a graph using a list of vertices and edges.
        """
        self.graph = graph

    def validityCheck(self, cover):
        """
        Check if a cover is valid by testing the conditions that would
        prove it invalid.
        """
        for i in range(len(self.graph)):
            for j in range(i+1, len(self.graph[i])): # check em
                if self.graph[i][j] == 1 and cover[i] != "1" and cover[j] != "1": # check coverage
                    return False
        return True
    
    def naiveSearch(self):
        """
        Search for vertex covers.
        """ 
