
"""
Kruskal's (Kruskal) algorithm finds a safe edge to add to the growing
forest by finding an edge of least weight.
"""

def union(p, r, x, y):
    """
    Return the union of two sets x and y with rank r and parents p
    """

def kruskal(g, w)
    """
    For graph g (tuple of vertices and edges) and weights w.
    """
    result = [] # resulting minimum spanning tree (MST) 
    i = 0 # sorted edges  
    e = 0 
    g = sorted(g, key=lambda i:w[i]) # sort by weight
    parent = []
    rank = []
    for n in g[1]: # for each vertex node
         parent.append(n)
         rank.append(0)
    while e < v[1]-1:
         u, v, w = g[0][i]
         i += 1
         x = find(parent, u)
         y = find(parent, v)
         if x != y:
             e += 1
             result.append([u, v, w])
             union(parent, rank, x, y)
    for u, v, w, in result:
         print("%d %d %d" % (u, v, w))
