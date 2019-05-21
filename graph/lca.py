"""
Lowest common ancestor (lca) for a given pair of nodes.
"""

def lca(G, node1, node2, default=None):
    """
    Compute the lowest common ancestor of the given pair of nodes.
    Only defined on non-null directed acyclic graphs.
    Takes n log(n) time in the size of the graph.
    """
