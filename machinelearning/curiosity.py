import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt

"""
Curiosity-driven learning (curiosity) implementation. 
"""

class CuriosityNet:
    def __init__(
            self,
            n_a,
            n_s,
            lr=0.01,
            gamma=0.98,
            epsilon=0.95,
            replace_target_iter=300,
            memory_size=10000,
            batch_size=128,
            output_graph=False,
    ):
        self.n_a = n_a
        self.n_s = n_s
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
