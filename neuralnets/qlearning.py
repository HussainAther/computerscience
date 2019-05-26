import gym 
import itertools 
import matplotlib 
import matplotlib.style 
import numpy as np 
import pandas as pd 
import sys
import plotting

from collections import defaultdict
from windy_gridworld import WindyGridworldEnv

# Set matplotlib style
matplotlib.style.use("ggplot")

# Create gym environment
env = WindyGridworldEnv()  

"""
The Q in Q-learning represents the expected utility (Quality) of a state-action 
pairing. Q-learning models learn about state-action pairs, such that for 
any given state, multiple possible actions may be entertained. (q learning qlearning Qlearning). 
"""

def createEpsilonGreedyPolicy(Q, epsilon, num_actions): 
    """ 
    Creates an epsilon-greedy policy based 
    on a given Q-function and epsilon. 
       
    Returns a function that takes the state 
    as an input and returns the probabilities 
    for each action in the form of a numpy array  
    of length of the action space(set of possible actions). 
    """
    def policyFunction(state): 
