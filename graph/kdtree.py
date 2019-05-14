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

class Orthotope(object):
    """
    k-dimensional rectangle.
    """
    __slots__ = ("min", "max")
 
    def __init__(self, mi, ma):
        """
        Min and max for the orthotope.
        """
        self.min, self.max = mi, ma

class KdTree(object):
    """
    Initialize the k-dimensional tree.
    """
    __slots__ = ("n", "bounds")
    def __init__(self, pts, bounds):
        """
        Use the points and boundaries to create the tree.
        """
        def nk2(split, exset):
            """
            Split along an exset to create the tree.
            """
            if not exset:
                return None
            exset.sort(key=itemgetter(split))
            m = len(exset) // 2
            d = exset[m]
            while m + 1 < len(exset) and exset[m + 1][split] == d[split]:
                m += 1
            s2 = (split + 1) % len(d)  # cycle coordinates
            return KdNode(d, split, nk2(s2, exset[:m]),
                                    nk2(s2, exset[m + 1:]))
        self.n = nk2(0, pts)
        self.bounds = bounds

T3 = namedtuple("T3", "nearest dist_sqd nodes_visited")

def find_nearest(k, t, p):
    """
    For target p, kd tree t, k number of nodes, find the nearest nodes 
    using the nn function.  
    """
    def nn(kd, target, hr, max_dist_sqd):
        if kd is None:
            return T3([0.0] * k, float("inf"), 0)
        nodes_visited = 1
        s = kd.split
        pivot = kd.dom_elt
        left_hr = deepcopy(hr)
        right_hr = deepcopy(hr)
        left_hr.max[s] = pivot[s]
        right_hr.min[s] = pivot[s]
        if target[s] <= pivot[s]:
            nearer_kd, nearer_hr = kd.left, left_hr
            further_kd, further_hr = kd.right, right_hr
        else:
            nearer_kd, nearer_hr = kd.right, right_hr
            further_kd, further_hr = kd.left, left_hr
        n1 = nn(nearer_kd, target, nearer_hr, max_dist_sqd)
        nearest = n1.nearest
        dist_sqd = n1.dist_sqd
        nodes_visited += n1.nodes_visited
        if dist_sqd < max_dist_sqd:
            max_dist_sqd = dist_sqd
        d = (pivot[s] - target[s]) ** 2
        if d > max_dist_sqd:
            return T3(nearest, dist_sqd, nodes_visited)
        d = sqd(pivot, target)
        if d < dist_sqd:
            nearest = pivot
            dist_sqd = d
            max_dist_sqd = dist_sqd
  
