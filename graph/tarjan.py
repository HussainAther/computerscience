import networkx as nx

from collections import defaultdict
from collections.abc import Mapping, Set
from itertools import chain, count
from networkx.utils import arbitrary_element, not_implemented_for, UnionFind, generate_unique_node

"""
Least common ancestors (lca) using Tarjan's algorithm. (tarjan Tarjan)
"""

def tarjan(G, root=None, pairs=None):
    """
    LCA for graph G.
    """
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("LCA meaningless on null graphs.")
    elif None in G:
        raise nx.NetworkXError("None is not a valid node.")

    # Index pairs of interest for efficient lookup from either side.
    if pairs is not None:
        pair_dict = defaultdict(set)
        # See note on all_pairs_lowest_common_ancestor.
        if not isinstance(pairs, (Mapping, Set)):
            pairs = set(pairs)
        for u, v in pairs:
            for n in (u, v):
                if n not in G:
                    msg = "The node %s is not in the digraph." % str(n)
                    raise nx.NodeNotFound(msg)
            pair_dict[u].add(v)
            pair_dict[v].add(u)
    if root is None:
        for n, deg in G.in_degree:
            if deg == 0:
                if root is not None:
                    msg = "No root specified and tree has multiple sources."
                    raise nx.NetworkXError(msg)
                root = n
            elif deg > 1:
                msg = "Tree LCA only defined on trees; use DAG routine."
                raise nx.NetworkXError(msg)
    if root is None:
        raise nx.NetworkXError("Graph contains a cycle.")
    uf = UnionFind()
    ancestors = {}
    for node in G:
        ancestors[node] = uf[node]
    colors = defaultdict(bool)
    for node in nx.dfs_postorder_nodes(G, root):
        colors[node] = True
        for v in (pair_dict[node] if pairs is not None else G):
            if colors[v]:
                # If the user requested both directions of a pair, give it.
                # Otherwise, just give one.
                if pairs is not None and (node, v) in pairs:
                    yield (node, v), ancestors[uf[v]]
                if pairs is None or (v, node) in pairs:
                    yield (v, node), ancestors[uf[v]]
        if node != root:
            parent = arbitrary_element(G.pred[node])
            uf.union(parent, node)
            ancestors[uf[parent]] = parent

