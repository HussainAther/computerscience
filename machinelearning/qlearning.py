import gym 
import itertools 
import matplotlib as mpl
import numpy as np 
import pandas as pd 
import sys
import plotting

from collections import defaultdict
from windy_gridworld import WindyGridworldEnv

# Set matplotlib style
mpl.style.use("ggplot")

# Create gym environment
env = WindyGridworldEnv()  

"""
The Q in Q-learning represents the expected utility (Quality) of a state-action 
pairing. Q-learning models learn about state-action pairs, such that for 
any given state, multiple possible actions may be entertained. (q learning qlearning Qlearning). Q-learning 
is a type of reinforcement algorithm for learning. 
"""

def createEpsilonGreedyPolicy(Q, epsilon, num_actions): 
    """ 
    Creates an epsilon-greedy policy based 
    on a given Q-function and epsilon for a number of 
    actions num_actions. 
       
    Returns a function that takes the state 
    as an input and returns the probabilities 
    for each action in the form of a numpy array  
    of length of the action space(set of possible actions). 
    """
    def policyFunction(state):
        """
        Select a function based on the state for this policy.
        """
        Action_probabilities = np.ones(num_actions, 
                dtype = float) * epsilon / num_actions 
                  
        best_action = np.argmax(Q[state]) 
        Action_probabilities[best_action] += (1.0 - epsilon) 
        return Action_probabilities 
   
    return policyFunction 

# Build model
def qLearning(env, num_episodes, discount_factor = 1.0, alpha = 0.6, epsilon = 0.1): 
    """ 
    Q-Learning algorithm: Off-policy TD control. 
    Finds the optimal greedy policy while improving 
    following an epsilon-greedy policy.
    """
    # Action value function 
    # A nested dictionary that maps 
    # state -> (action -> action-value). 
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # Keeps track of useful statistics 
    stats = plotting.EpisodeStats( 
        episode_lengths = np.zeros(num_episodes), 
        episode_rewards = np.zeros(num_episodes))     
       
    # Create an epsilon greedy policy function 
    # appropriately for environment action space. 
    policy = createEpsilonGreedyPolicy(Q, epsilon, env.action_space.n) 
    # For every episode 
    for ith_episode in range(num_episodes): 
           
        # Reset the environment and pick the first action.
        state = env.reset() 
           
        for t in itertools.count(): 
            # get probabilities of all actions from current state 
            action_probabilities = policy(state) 
   
            # choose action according to  
            # the probability distribution 
            action = np.random.choice(np.arange( 
                      len(action_probabilities)), 
                       p = action_probabilities) 
   
            # take action and get reward, transit to next state 
            next_state, reward, done, _ = env.step(action) 
   
            # Update statistics 
            stats.episode_rewards[i_episode] += reward 
            stats.episode_lengths[i_episode] = t 
               
            # TD Update 
            best_next_action = np.argmax(Q[next_state])     
            td_target = reward + discount_factor * Q[next_state][best_next_action] 
            td_delta = td_target - Q[state][action] 
            Q[state][action] += alpha * td_delta 
   
            # done is True if episode terminated    
            if done: 
                break
           state = next_state 
       
    return Q, stats 

# Train the model
Q, stats = qLearning(env, 1000) 

# Plot
plotting.plot_episode_stats(stats) 

"""
Q-learning brain 
"""

class RL(object):
    """
    Reinforcement learning (RL) model class.
    """
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass

 class QLearningTable(RL):
    """
    Fill out a q_table with the target and prediction values.
    """
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != "terminal":
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

"""
SARSA is an on-policy algorithm where, in the current state, S an action, A is taken and the 
agent gets a reward, R and ends up in next state, S1 and takes action, A1 in S1. Therefore, 
the tuple (S, A, R, S1, A1) stands for the acronym SARSA.
"""

class SarsaTable(RL):
    """
    Fill out the SARSA table for our RL model.
    """
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        """
        Learning using SARSA.
        """
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != "terminal":
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
