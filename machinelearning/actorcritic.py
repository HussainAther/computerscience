import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil

"""
The actor-critic model is a two-part model with many of the same features as 
the Q-learning model. One component, the "Critic" is an evaluator of the state's 
value. The Critic learns the value of the stimulus without taking into account 
the possible actions. The second component, the "Actor," is used for action 
selection and learns stimulus-response weightings for each state- action pair 
as a function of the critic’s evaluation. PEs are generated at the Critic level 
to update both the state value of the critic and the stimulus-response weights of the Actor.

Asynchronous advantage actor critic (A3C a3c) reinforcement learning.
"""

GAME = "BipedalWalker-v2'
OUTPUT_GRAPH = False
LOG_DIR = "./log"
N_WORKERS = multiprocessing.cpu_count()
MAX_GLOBAL_EP = 8000
GLOBAL_NET_SCOPE = "Global_Net"
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.99
ENTROPY_BETA = 0.005
LR_A = 0.00005 # learning rate for actor
LR_C = 0.0001  # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

env = gym.make(GAME)

N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]
del env
