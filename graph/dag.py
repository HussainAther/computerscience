import matplotlib.pyplot as plt
import pylab
import networkx as nx
import numpy as np

from pylab import rcParams

"""
Directed acyclic graph.
"""

# Set graph display size as 10 x 10.
rcParams["figure.figsize"] = 10, 10

G = nx.DiGraph()

# Add edges and weights.
G.add_edges_from([("K", "I"), ("R", "T"), ("V", "T")], weight=3)
G.add_edges_from([("T", "K"), ("T", "H"), ("T", "H")], weight=4)

# These values determine node colors.
val_map = {"L": 1.5, "I": .9, "R": .6, "T":.2}
values = [val_map.get(node, 1.0) for node in G.nodes()]

# Label the edges.
edges_labels = dict([((u, v), d["weight"]) for u, v, d in G.edges(data=True)])

# Set edge colors.
red_edges = [("R", "T"), ("T", "K")]
edge_colors = ["green" if not edge in red_edges else "red" for edge in G.edges()]

pos = nx.spring_layout(G)

nx.draw_networkx_edges(G, pos, width=2.0, alpha=.65)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

nx.draw(G, pos, node_color = values, node_size=1500, edge_color=edge_colors, edge_cmap=plt.cm.Reds)

pylab.show()

nx.is_directed_acyclic_graph(G)
