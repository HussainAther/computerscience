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
"""
