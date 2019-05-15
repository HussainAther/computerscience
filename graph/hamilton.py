"""
The Hamiltonian path problem (hamiltonian)

We need an algorithm to find a path in a graph that visits every node exactly once, 
if such a path exists.
"""

class Graph:
    def __init__(self, vlist=[]):
        """ 
        Initialize a Graph with an optional vertex list. 
        """
        self.index = {v:i for i,v in enumerate(vlist)}
        self.vertex = {i:v for i,v in enumerate(vlist)}
        self.edge = []
        self.edgelabel = []
    def addVertex(self, label):
        """ 
        Add a labeled vertex to the graph. 
        """
        index = len(self.index)
        self.index[label] = index
        self.vertex[index] = label
    def addEdge(self, vsrc, vdst, label="", repeats=True):
        """ 
        Add a directed edge to the graph, with an optional label. 
        Repeated edges are distinct, unless repeats is set to False. 
        """
        e = (self.index[vsrc], self.index[vdst])
        if (repeats) or (e not in self.edge):
            self.edge.append(e)
            self.edgelabel.append(label)
