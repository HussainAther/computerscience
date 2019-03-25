
"""
Kruskal's (Kruskal) algorithm finds a safe edge to add to the growing
forest by finding an edge of least weight.
"""

def kruskal(g, w)
    """
    For graph g (tuple of vertices and edges) and weights w.
    """
    result = [] # resulting minimum spanning tree (MST) 
    i = 0 # sorted edges  
    e = 0 
    g = sorted(g, key=lambda i:w[i]) # sort by weight
 
    
