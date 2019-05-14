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
