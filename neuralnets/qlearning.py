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
