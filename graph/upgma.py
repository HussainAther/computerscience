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

def distmat(points):
    """
    Convert array of points points to distance matrix using distance measure.
    """
    n = len(points)
    dim = len(points[0])
    dist = [[0 for x in xrange(n)] for y in xrange(n)]
    for i in xrange(n):
        for j in xrange(n):
            d = 0
            p1 = points[i]
            p2 = points[j]
            for k in xrange(dim):
                d = d + (p1[k]-p2[k])**2
            dist[i][j] = math.sqrt(d)
    return dist

def euclidistance(c1, c2):
    """ 
    Calculate the distance between two clusters c1 and c2. 
    """
    dist = .0
    n1 = len(c1.points)
    n2 = len(c2.points)
    for i in xrange(n1):
        for j in xrange(n2):
            p1 = c1.points[i]
            p2 = c2.points[j]
            dim = len(p1)
            d = 0
            for k in xrange(dim):
                d = d + (p1[k]-p2[k])**2
            d = math.sqrt(d)
            dist = dist + d
    dist = dist / (n1*n2)
    return dist

def upgma(points, k):
    """ 
    Cluster based on distance matrix dist using Unweighted Pair Group 
    Method with Arithmetic Mean algorithm up to k cluster.
    """
    # Initialize each cluster with one point
    nodes = []
    n = len(points)
    for i in xrange(n):
        node = Node([points[i]])
        nodes = nodes + [node]
    # Iterate until the number of clusters is k
    nc = n
    while nc > k:
        # Calculate the pairwise distance of each cluster, while searching for pair with least distance
        c1 = 0; c2 = 0; i1 = 0; i2 = 0;
        sdis = 9999999999 # big number to minimize distance measure
        for i in xrange(nc):
            for j in xrange(i+1, nc):
                dis = euclidistance(nodes[i], nodes[j])
                if dis < sdis:
                    sdis = dis
                    c1 = nodes[i]; c2 = nodes[j];
                    i1 = i; i2 = j;
        # Merge these two nodes into one new node
        node = Node(c1.points + c2.points)
        node.left = c1; node.right = c2;
