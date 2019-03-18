
"""
You have gotten into trouble with some of your, shall we say, difficult friends because
they have realized that they are not being invited to your house parties. So you announce
that you are going to have dinners on two consecutive nights, Friday and Saturday, and
every one of your friends is going to get invited on exactly one of those two days. You
are still worried about pairs of your friends who intensely dislike each other and want to
invite them on different days.

Now, you are a bit worried because you don’t know if you can pull this off. If you
had a small social circle like the one below, it would be easy. Recall that in the
graph below, the vertices are friends and an edge between a pair of vertices mean
that the two corresponding friends dislike each other, and cannot be invited to the
party on the same night.

To summarize:
1. Each of your friends must attend exactly one of the two dinners.
2. If A dislikes B or B dislikes A, they cannot both be in the same dinner party.

Can you invite all of your friends A through I above following your self-imposed rules?
Or are you going to have go to back on your word, and not invite someone?
"""

dgraph = { 'B': ['C'],
           'C': ['B', 'D'],
           'D': ['C', 'E', 'F'],
           'E': ['D'],
           'F': ['D', 'G', 'H', 'I'],
           'G': ['F'],
           'H': ['F'],
           'I': ['F'],
          'F1': ['D1', 'I1', 'G1', 'H1'],
          'B1': ['C1'],
          'D1': ['C1', 'E1', 'F1'],
          'E1': ['D1'],
          'H1': ['F1'],
          'C1': ['D1', 'B1'],
          'G1': ['F1'],
          'I1': ['F1']}

dgraph2 = {'F': ['D', 'I', 'G', 'H'],
           'B': ['C'],
           'D': ['C', 'E', 'F'],
           'E': ['D'],
           'H': ['F'],
           'C': ['D', 'B'],
           'G': ['F'],
           'I': ['F'],
           'A1': ['B1', 'C1'],
           'B1': ['A1', 'C1'],
           'C1': ['A1', 'B1']}

dgraph3 = {'A': ['B'],
           'B': ['A'],
           'C': ['D'],
           'D': ['C', 'E', 'F'],
           'E': ['D'],
           'F': ['D', 'G', 'H', 'I'],
           'G': ['F'],
           'H': ['F'],
           'I': ['F']}

def bipartiteGraphColor(graph, start, coloring, color):
    """
    Color the graph.
    """
    if start not in graph:
        return False, {}
    
    if start not in coloring:
        coloring[start] = color
    elif coloring[start] != color:
        return False, {}
    else:
        return True, coloring

    if color == "Sha":
        newcolor = "Hat"
    else:
        newcolor = "Sha"

    for vertex in graph[start]:
        val, coloring = bipartiteGraphColor(graph, vertex, coloring, newcolor)
        if val == False:
            return False, {}

    return True, coloring

def colorDisconnectedGraph(graph, coloring):
    """
    Color each vertex.
    """
    for g in graph:
        if g not in coloring:
            success, coloring = bipartiteGraphColor(graph, g, coloring, 'Sha')
            if not success:
                return False, {}

    return True, coloring
        
