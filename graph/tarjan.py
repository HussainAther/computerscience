from collections import defaultdict
from collections.abc import Mapping, Set
from itertools import chain, count

import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for, UnionFind, generate_unique_node

"""
Least common ancestors (lca) using Tarjan's algorithm. (tarjan Tarjan)
"""

def tarjan(G, root=None, pairs=None):
