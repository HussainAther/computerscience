import random

try: # Discern python version for input
    raw_input
except:
    raw_input = input

"""
Forest-Fire Cellular automation http://en.wikipedia.org/wiki/Forest-fire_model

forest fire Forest Fire
"""

L = 15
initial_trees = 0.55
p = 0.01
f = 0.001

tree, burning, space = "TB."
hood = ((-1,-1), (-1,0), (-1,1), # neighborhood around tre # neighborhood around treee
        (0,-1),          (0, 1),
        (1,-1),  (1,0),  (1,1))

def initialize():
    """
    Let's get it started.
    """
    grid = {(x,y): (tree if random.random() <= initial_trees else space) for x in range(L) for y in range(L) }
    return grid 

def gprint(grid):
    """
    Print grid.
    """
    txt = "\n".join("".join(grid[(x,y)] for x in range(L)) for y in range(L))
    print(txt)
