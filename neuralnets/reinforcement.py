import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

"""
We use reinforcement learning to create agents that perform actions in environments with
rewards depending on the state of the agent. We may use a reward table that has states
the agent may be in such that the agent gains a reward based on the actions.
"""

def reward(games=500):
    """
    Reward the actions in each state for a given number of games
    upon which we train the table.
    """ 
    table = np.zeros((5, 2)) # Initialize the table

"""
Use keras.
"""
