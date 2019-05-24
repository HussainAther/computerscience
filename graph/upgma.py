import math
import matplotlib.pyplot as plt

"""
UPGMA (unweighted pair group method with arithmetic mean) is a simple 
agglomerative (bottom-up) hierarchical clustering method. 
"""

class Node:
    """
    Basic Node node class for the tree nodes.
    """
    def __init__(self, p):
        self.points = p
        self.right = None
        self.left = None

