"""
Dijkstra's algorithm finds the shortest paths between nodes in a graph.
For a given source node, we find the shortest path between that node and every other.
We can use it to find the shortest paths from a single node to a single destination by
stopping the algorithm once the shortest path has been determined
"""

class Graph:

    def __init__(self):
        """
        This class is for simple book-keeping
        of the lists of edges and nodes.
        """
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}
    
    def add_node(self, value):
	"""
	Used to add each node to the graph.
	"""
        self.nodes.add(value)

    def add_edge(self, from_node, to_node, distance):
	"""
	Keep track of edges.
	"""
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.distances[(from_node, to_node)] = distance

def dijsktra(graph, initial):
    """
    Dijkstra's algorithm to find the shortest path between a and b.
    It picks the unvisited vertex with the lowest distance, calculates
    the distance through it to each unvisited neighbor, and updates the
    neighbor's distance if smaller.
    """
    visited = {initial: 0}
    path = {}

    nodes = set(graph.nodes)

    while nodes:
        min_node = None
        for node in nodes:
            if node in visited:
                if min_node is None:
                    min_node = node
                elif visited[node] < visited[min_node]:
                    min_node = node

    if min_node is None:
        break

    nodes.remove(min_node)
    current_weight = visited[min_node]

    for edge in graph.edges[min_node]:
        weight = current_weight + graph.distance[(min_node, edge)]
            if edge not in visited or weight < visited[edge]:
                visited[edge] = weight
                path[edge] = min_node

    return visited, path
