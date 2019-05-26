import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil

"""
The actor-critic model is a two-part model with many of the same features as 
the Q-learning model. One component, the "Critic" is an evaluator of the state"s
value. The Critic learns the value of the stimulus without taking into account 
the possible actions. The second component, the "Actor," is used for action 
selection and learns stimulus-response weightings for each state- action pair 
as a function of the critic’s evaluation. PEs are generated at the Critic level 
to update both the state value of the critic and the stimulus-response weights of the Actor.

Asynchronous advantage actor critic (A3C a3c) reinforcement learning.
"""

GAME = "BipedalWalker-v2"
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

class ACNet(object):
    """
    Create a network from which the actor and critic can use.
    """
    def __init__(self, scope, globalAC=None):
        """
        Initialize the network and its scope.
        """
        if scope == GLOBAL_NET_SCOPE: # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], "S")
                self._build_net()
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + "/actor")
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + "/critic")
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], "S")
                self.a_his = tf.placeholder(tf.float32, [None, N_A], "A")
                self.v_target = tf.placeholder(tf.float32, [None, 1], "Vtarget")
                mu, sigma, self.v = self._build_net()
                td = tf.subtract(self.v_target, self.v, name="TD_error")
                with tf.name_scope("c_loss"):
                    self.c_loss = tf.reduce_mean(tf.square(td))
                with tf.name_scope("wrap_a_out"):
                    self.test = sigma[0]
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-5
                normal_dist = tf.contrib.distributions.Normal(mu, sigma)
                with tf.name_scope("a_loss"):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * td
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)
             with tf.name_scope("choose_a"):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1)), *A_BOUND)
                with tf.name_scope("local_grad"):
                    self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + "/actor")
                    self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + "/critic")
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)
            with tf.name_scope("sync"):
                with tf.name_scope("pull"):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope("push"):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self):
        """
        Build a network.
        """
        w_init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("actor"):
            l_a = tf.layers.dense(self.s, 500, tf.nn.relu6, kernel_initializer=w_init, name="la")
            l_a = tf.layers.dense(l_a, 300, tf.nn.relu6, kernel_initializer=w_init, name="la2")
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name="mu")
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name="sigma")
        with tf.variable_scope("critic"):
            l_c = tf.layers.dense(self.s, 500, tf.nn.relu6, kernel_initializer=w_init, name="lc")
            l_c = tf.layers.dense(l_c, 300, tf.nn.relu6, kernel_initializer=w_init, name="lc2")
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name="v")  # state value
        return mu, sigma, v

    def update_global(self, feed_dict):  # run by a local
        _, _, t = SESS.run([self.update_a_op, self.update_c_op, self.test], feed_dict)  # local grads applies to global net
        return t

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        return SESS.run(self.A, {self.s: s})

class Worker(object):
    def __init__(self, name, globalAC):
        self.env = gym.make(GAME)
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            while True:
                if self.name == 'W_0' and total_step % 30 == 0:
                    self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                if r == -100: r = -2

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    test = self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()
