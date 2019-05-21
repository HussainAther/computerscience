
"""
Lowest common ancestor (lca) for a given pair of nodes.
"""

def allpairslca(G, pairs=None):
    """
    Compute the lowest common ancestor for pairs of nodes.
    """
    if not nx.is_directed_acyclic_graph(G):
        raise nx.NetworkXError("LCA only defined on directed acyclic graphs.")
    elif len(G) == 0:
        raise nx.NetworkXPointlessConcept("LCA meaningless on null graphs.")
    elif None in G:
        raise nx.NetworkXError("None is not a valid node.")
    if (not isinstance(pairs, (Mapping, Set)) and pairs is not None):
        pairs = set(pairs)
    sources = [n for n, deg in G.in_degree if deg == 0]
    if len(sources) == 1:
        root = sources[0]
        super_root = None
    else:
        G = G.copy()
        super_root = root = generate_unique_node()
        for source in sources:
            G.add_edge(root, source)
    spanning_tree = nx.dfs_tree(G, root)
    dag = nx.DiGraph((u, v) for u, v in G.edges
                     if u not in spanning_tree or v not in spanning_tree[u])


def lca(G, node1, node2, default=None):
    """
    Compute the lowest common ancestor of the given pair of nodes.
    Only defined on non-null directed acyclic graphs.
    Takes n log(n) time in the size of the graph.
    """
    ans = list(allpairslca(G, pairs=[(node1, node2)]))
    if ans:
        assert len(ans) == 1
        return ans[0][1]
    else:
        return default
