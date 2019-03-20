import numpy as np
import pandas as pd
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib as mpl

"""
Voronoi diagrams let us divide spcae into several regions such that each region
has all the points lcoser to one point than to any other seed point.
"""

def voronoi(vor, radius=None):
    """
    Create infinite Voronoi regions in a 2D diagram to finite regions.
    """
