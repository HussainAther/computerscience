from collections import defaultdict
from collections.abc import Mapping, Set
from itertools import chain, count

import networkx as nx
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
