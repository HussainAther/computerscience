from collections import deque, namedtuple

Vertex = namedtuple('Vertex', ['name', 'incoming', 'outgoing'])

"""
Kahn's (Kahn) algorithm for topological sort takes a directed acyclic graph 
with a linera ordering of the vertices such that for every directed edge ev, vertex u comes
before v in the ordering. A directed acyclic graph is a finite directed graph with no
directed cycles. A directed graph is a graph that is made of vertices connected by edges
in which the edges have a direct association with them. A directed cycle is a directed
version of a cycle graph with all edges in the same direction.
"""

def build_doubly_linked_graph(graph):
    """
    Given a graph with only outgoing edges, build a graph with incoming and
    outgoing edges. The returned graph will be a dictionary mapping vertex to a
    Vertex namedtuple with sets of incoming and outgoing vertices.
    """
    g = {v:Vertex(name=v, incoming=set(), outgoing=set(o)) for v, o in graph.items()}
    for v in g.values():
        for w in v.outgoing:
            if w in g:
                g[w].incoming.add(v.name)
            else: 
                g[w] = Vertex(name=w, incoming={v}, outgoing=set())
    return g


def kahn_top_sort(graph):
    """
    Given an acyclic directed graph return a
    dictionary mapping vertex to sequence such that sorting by the sequence
    component will result in a topological sort of the input graph. Output is
    undefined if input is a not a valid DAG.

    The graph parameter is expected to be a dictionary mapping each vertex to a
    list of vertices indicating outgoing edges. For example if vertex v has
    outgoing edges to u and w we have graph[v] = [u, w].
    """
    g = build_doubly_linked_graph(graph)
    # sequence[v] < sequence[w] implies v should be before w in the topological
    # sort.
    q = deque(v.name for v in g.values() if not v.incoming)
    sequence = {v: 0 for v in q}
    while q:
        v = q.popleft()
        for w in g[v].outgoing:
            g[w].incoming.remove(v)
        if not g[w].incoming:
            sequence[w] = sequence[v] + 1
            q.append(w)
    return sequence
