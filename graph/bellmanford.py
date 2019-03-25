from collections import defaultdict

"""
The Bellman-Ford algorithm for computing shortest paths from a
single source vertex to all the other vertices in a weighted graph
is slower that Dijkstra's algorithm, but more versatile and even
handling edge weights of negative numbers. It's also sometimes called
the Bellman-Ford-Moore algorithm.
"""

def addEdge(self,u,v,w):
    self.graph.append([u, v, w])

def BellmanFord(graph, src):

    dist = [float("Inf")] * self.V # initialize distances from the source
    dist[src] = 0

    for i in range(self.V - 1):
        for u, v, w in graph:
            if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
    for u, v, w in graph:
            if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                    # Negative weight cycle
                    return

    # print all distance
    self.printArr(dist)

V= vertices #No. of vertices
graph = [] # default dictionary to store graph

for i in range(V):
    print("%d \t\t %d" % (i, dist[i]))
