import numpy as np
import networkx as nx
"""
We use the Barabasi-Albert model (BA barabasi albert Barabási) to explain power-law degree distributino
of networks by considering growth and preferential attachment. The BA algorithm in the BA model
starts (growth) with a small number m of connected nodes and, at every step, adds a new node with n<m edges
that link the new node ot m different nodes already present in the network. Then it performs preferential
attachment by choosing the nodes to which the new node connects based on some probability P that
the ne wnode will be connected to the node:

P ~ ki / summation for all i of ki

in which ki is the degree of node i.
"""

"""
We can use nx
"""
G = nx.barabasi_albert_graph(50,30)
nx.draw(G)

"""
Or implement it in Python.
"""

def ba(n, m):
    """
    Return a random graph according to the Barabási-Albert Attachment model for n 
    number of nodes and m number of edges to attach.
    """
    np.random.seed(1234) # or change this whatever
    g = []
    t = list(range(m) # target nodes for new edges
    n  = [] # list of existing nodes
    s = m # first node
    while s < n:
        g.append(m, t) 
