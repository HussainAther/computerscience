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
