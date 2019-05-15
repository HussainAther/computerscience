import itertools

"""
The Hamiltonian path problem (hamiltonian)

We need an algorithm to find a path in a graph that visits every node exactly once, 
if such a path exists. We use recursion to extend paths along edges that include
unvisited vertices. If we use all vertices, we've found a path. 
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
    def hamiltonianPath(self):
        """ 
        A Brute-force (brute Brute) method for finding a Hamiltonian Path. 
        Basically, all possible N! paths are enumerated and checked
        for edges. Since edges can be reused there are no distictions
        made for *which* version of a repeated edge. 
        """
        for path in itertools.permutations(sorted(self.index.values())):
            for i in xrange(len(path)-1):
                if ((path[i],path[i+1]) not in self.edge):
                    break
            else:
                return [self.vertex[i] for i in path]
        return []
    def SearchTree(self, path, verticesLeft):
        """ 
        A recursive Branch-and-Bound Hamiltonian Path search. 
        Paths are extended one node at a time using only available
        edges from the graph. 
        """
        if (len(verticesLeft) == 0):
            self.PathV2result = [self.vertex[i] for i in path]
            return True
        for v in verticesLeft:
            if (len(path) == 0) or ((path[-1],v) in self.edge):
                if self.SearchTree(path+[v], [r for r in verticesLeft if r != v]):
                    return True
        return False
    def hamiltonianPathV2(self):
        """ 
        A wrapper function for invoking the Branch-and-Bound 
        Hamiltonian Path search. 
        """
        self.PathV2result = []
        self.SearchTree([],sorted(self.index.values()))                
        return self.PathV2result
   def eulerianPath(self):
        """
        Eulerian (eulerian) cycle starts and ends on same vertex.
        """
        graph = [(src,dst) for src,dst in self.edge]
        currentVertex = self.verifyAndGetStart()
        path = [currentVertex]
        # "next" is where vertices get inserted into our tour
        # it starts at the end (i.e. it is the same as appending),
        # but later "side-trips" will insert in the middle
        next = 1
        while len(graph) > 0:
            # follows a path until it ends
            for edge in graph:
                if (edge[0] == currentVertex):
                    currentVertex = edge[1]
                    graph.remove(edge)
                    path.insert(next, currentVertex)
                    next += 1
                    break
            else:
                # Look for side-trips along the path
                for edge in graph:
                    try:
                        # insert our side-trip after the
                        # "u" vertex that is starts from
                        next = path.index(edge[0]) + 1
                        currentVertex = edge[0]
                        break
                    except ValueError:
                        continue
                else:
                    print("There is no path!")
                    return False
        return path

G1 = Graph(binary)
for vsrc in binary:
    G1.addEdge(vsrc,vsrc[1:]+"0")
    G1.addEdge(vsrc,vsrc[1:]+"1")
# This takes about 30 mins
%time path = G1.hamiltonianPath()
print(path)
