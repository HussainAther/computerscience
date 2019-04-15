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

class Agent(object):
    mode = "random"

    """
    Agent functions and actions.
    """

    def __init__(self, row, column, color=0):
        self.row = row
        self.col = column
        self.color = color
        self.next_steps = []

    def nearest_enemy(self, agents):
        agents = [a for a in agents if a.mode != self.mode]
        distances = [manhattan((agent.row, agent.col), (self.row, self.col))
                     for agent in agents]

        if not distances:
            return None

        md = np.argmin(distances)
        if distances[md] <= D:
            return agents[md]
        else:
            return None

    def get_steps(self, agents):
        steps = [((self.row + r) % P, (self.col + c) % Q)
                 for (r, c) in directions]
        steps.append((self.row, self.col))

        return steps

    def plan(self, agents):
        self.next_steps = self.get_steps(agents)

    def step(self, filled_positions):
        viable_steps = [pos for pos in self.next_steps
                      if not pos in filled_positions]

        try:
            next_row, next_col = random.choice(viable_steps)
        except IndexError:
            next_row, next_col = self.row, self.col

        filled_positions.remove((self.row, self.col))
        self.row = next_row
        self.col = next_col
        filled_positions.append((self.row, self.col))

        return filled_positions

class SpotterAgent(Agent):
    mode = "spotter"

    """
    Spotter agents run away. 
    """

    def get_steps(self, agents):
        possible_steps = Agent.get_steps(self, agents)
        enemy = self.nearest_enemy(agents)

        # Attack or run if there is an enemy nearby, otherwise
        # move randomly in any direction.
        if enemy is None:
            viable_steps = possible_steps
        else:
            distance_to_enemy = \
                       [manhattan((enemy.row, enemy.col), (step_r, step_c))
                        for (step_r, step_c) in possible_steps]

            if self.mode == 'hunt':
                dist_func = min
            else:
                dist_func = max

            viable_steps = [step for (step, dist) in
                            zip(possible_steps, distance_to_enemy)
                            if dist == dist_func(distance_to_enemy)]

        return viable_steps

class Chaser(SpotterAgent):
    mode = "hunt"
 
    """
    Chase others.
    """

    def __init__(self, row, column):
        Agent.__init__(self, row, column, color=1)

class Target(SpotterAgent):
    mode = "hide"

    """
    Target another agent.  
    """

    def __init__(self, row, column):
        Agent.__init__(self, row, column, color=2)

def killed(targets, chaser_positions):
    """
    RIP in peace.
    """
    dead_targets = []
    for t in targets:
        for (r, c) in chaser_positions:
            if (t.row == r) and (t.col == c):
                dead_targets.append(t)
                continue

    return dead_targets
