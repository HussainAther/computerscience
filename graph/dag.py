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
