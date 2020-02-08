# python3
import queue

"""
A tornado is approaching the city, and we need to evacuate the people quickly. There are several
roads outgoing from the city to the nearest cities and other roads going further. The goal is to evacuate
everybody from the city to the capital, as it is the only other city which is able to accomodate that
many newcomers. We need to evacuate everybody as fast as possible, and your task is to find out
what is the maximum number of people that can be evacuated each hour given the capacities of all
the roads.
"""

class edge:

    def __init__(self, u, v, c):
        self.u = u
        self.v = v
        self.c = c
        self.flow = 0
class flow:
    def __init__(self, n):
        self.edges = []
        self.graph = [[] for _ in range(n)]

    def add(self, f_, to, c):
        fe = edge(f_, to, c)
        be = edge(to, f_, 0)
        self.graph[f_].append(len(self.edges))
        self.edges.append(fe)
        self.graph[to].append(len(self.edges))
        self.edges.append(be)

    def size(self):
        return len(self.graph)

    def get_ids(self, f_):
        return self.graph[f_]

    def get_edge(self, id):
        return self.edges[id]

    def add_flow(self, id, flow):
        self.edges[id].flow += flow
        self.edges[id ^ 1].flow -= flow
        self.edges[id].c -= flow
        self.edges[id ^ 1].c += flow

def datain():
    vcount, ecount = map(int, input().split())
    graph = flow(vcount)
    for _ in range(ecount):
        u, v, c = map(int, input().split())
        graph.add(u - 1, v - 1, c)
    return graph


def maxflow(graph, f_, to):
    flow = 0
    while True:
        hasPath, path, X = bfs(graph, f_, to)
        if not hasPath:
            return flow
        for id in path:
            graph.add_flow(id, X)
        flow += X
    return flow

def bfs(graph, f_, to):
    X = float("inf")
    hasPath = False
    n = graph.size()
    dist = [float("inf")]*n
    path = []
    parent = [(None, None)]*n
    q = queue.Queue()
    dist[f_] = 0
    q.put(f_)
    while not q.empty():
        cf = q.get()
        for id in graph.get_ids(cf):
            ce = graph.get_edge(id)
            if float("inf") == dist[ce.v] and ce.c > 0:
                dist[ce.v] = dist[cf] + 1
                parent[ce.v] = (cf, id)
                q.put(ce.v)
                if ce.v == to:
                    while True:
                        path.insert(0, id)
                        c = graph.get_edge(id).c
                        X = min(c, X)
                        if cf == f_:
                            break
                        cf, id = parent[cf]
                    hasPath = True
                    return hasPath, path, X
    return hasPath, path, X

if __name__ == "__main__":
    graph = datain()
    print(maxflow(graph, 0, graph.size() - 1))