import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

import random

"""
Simple chase game.
"""

# Board size
P, Q = 80, 150

# Number of chasers and targets
N_c = 30
N_t = 100

# Detection distance
D = np.infty

directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
board = np.zeros((P, Q), dtype=np.uint8)

def manhattan((r0, c0), (r1, c1)):
    """
    Wrapped manhattan distance between two points on the grid.
    """
    if r0 > r1:
        r0, r1 = r1, r0

    if c0 > c1:
        c0, c1 = c1, c0

    row_dist = min(r1 - r0, r0 + (P - r1))
    col_dist = min(c1 - c0, c0 + (Q - c1))

    return row_dist + col_dist

