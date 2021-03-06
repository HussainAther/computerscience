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

def act_and_draw(N, image, board, agents):
    # Shuffle agents before each step of the simulation
    random.shuffle(agents)

    # Run one step of the simulation
    for a in agents:
        a.plan(agents)

    chasers = [a for a in agents if a.mode == "hunt"]
    targets = [a for a in agents if a.mode != "hunt"]

    filled_positions = [(a.row, a.col) for a in chasers]
    for a in chasers:
        filled_positions = a.step(filled_positions)

    for t in killed(targets, filled_positions):
        agents.remove(t)

    filled_positions.extend((a.row, a.col) for a in targets)
    for a in targets:
        filled_positions = a.step(filled_positions)

    # Draw agents
    board.fill(0)
    for a in agents:
        board[a.row, a.col] = a.color

    # Update board
    image.set_data(board)

    # Reset filled_positions
    filled_positions = []

    ## Uncomment to make video frames
    # plt.savefig('/tmp/chase_escape_%05d.png' % N)

    return image,

if __name__ == "__main__":
    # Initial placement of agents.
    #
    # Take all positions, shuffle them, and randomly pick
    # the necessary number.
    #
    positions = list(np.ndindex(P, Q))
    random.shuffle(positions)
    positions_c = positions[:N_c]
    positions_t = positions[N_c:N_c + N_t]

    # Construct agents based on initial positions
    agents = [Chaser(r, c) for (r, c) in positions_c]
    agents.extend([Target(r, c) for (r, c) in positions_t])

    # Set up display
    fig = plt.figure(figsize=(10, 10 * (P / float(Q))))
    cmap = ListedColormap(['k', 'r', 'g'])
    image = plt.imshow(board, interpolation='nearest',
                       cmap=cmap, vmin=0, vmax=2)

    # Animate
    anim = FuncAnimation(fig, func=act_and_draw,
                         fargs=(image, board, agents), interval=0, blit=True)

    plt.show()

