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
        # total learning step
        self.learn_step_counter = 0
        self.memory_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_s * 2 + 2))
        self.tfs, self.tfa, self.tfr, self.tfs_, self.dyn_train, self.dqn_train, self.q, self.int_r = \
            self._build_nets()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_nets(self):
        tfs = tf.placeholder(tf.float32, [None, self.n_s], name="s") # input State
        tfa = tf.placeholder(tf.int32, [None, ], name="a") # input Action
        tfr = tf.placeholder(tf.float32, [None, ], name="ext_r") # extrinsic reward
        tfs_ = tf.placeholder(tf.float32, [None, self.n_s], name="s_")  # input Next State

        # dynamics net
        dyn_s_, curiosity, dyn_train = self._build_dynamics_net(tfs, tfa, tfs_)

        # normal RL model
        total_reward = tf.add(curiosity, tfr, name="total_r")
        q, dqn_loss, dqn_train = self._build_dqn(tfs, tfa, total_reward, tfs_)
        return tfs, tfa, tfr, tfs_, dyn_train, dqn_train, q, curiosity
