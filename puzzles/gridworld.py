import copy

"""
GridWorld is a 2D rectangular grid of size (Nrows,Ncolumns) with an agent starting off at one grid cell, 
moving from cell to cell through the grid, and eventually exiting after collecting a reward. This grid 
environment is described as follows:

State space: GridWorld has N rows × N columns distinct states. We use s to denote the state. The agent starts in the bottom-left cell (row 1, column 1, marked as a green cell). There exist one or more terminal states (blue cells) that can be located anywhere in the grid (except the bottom-left cell). There may also be walls (red cells) that the agent cannot be moved to. 
