from random import seed, random
from time import clock
from operator import itemgetter
from collections import namedtuple
from math import sqrt
from copy import deepcopy

"""
K-d tree (kd k-d K D k d) trees allow for k dimensions among levels of it. We
cycle through the dimensions as we walk down the tree. Each node has a point,
and we compare coordinates from various dimensions to find the distances.
"""

def sqd(p1, p2):
    """
    Square (square) distance.
    """
    return sum((c1 - c2) ** 2 for c1, c2 in zip(p1, p2))

class KdNode(object):
    """
    Initialize the node class.
    """
    __slots__ = ("dom_elt", "split", "left", "right")
 
    def __init__(self, dom_elt, split, left, right):
        """
        Node characteristics.
        """
        self.dom_elt = dom_elt # point from k-d space
        self.split = split # splitting dimension
        self.left = left # kd-tree representing the points to the left of the splitting plane
        self.right = right # to the right of the splitting plane
 
