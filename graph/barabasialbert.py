import numpy as np

"""
We use the Barabasi-Albert model (BA barabasi albert) to explain power-law degree distributino
of networks by considering growth and preferential attachment. The BA algorithm in the BA model
starts (growth) with a small number m of connected nodes and, at every step, adds a new node with n<m edges
that link the new node ot m different nodes already present in the network. Then it performs preferential
attachment by choosing the nodes to which the new node connects based on some probability P that
the ne wnode will be connected to the node:

P ~ ki / summation for all i of ki

in which ki is the degree of node i.
"""
