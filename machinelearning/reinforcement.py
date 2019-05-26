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
Use keras. We will be using Deep Q-learning algorithm. Q-learning is a 
policy based learning algorithm with the function approximator as a neural 
network. This algorithm was used by Google to beat humans at Atari games.
"""

ENV_NAME = "CartPole-v0"

"""
Get the environment and extract the number of actions available in the Cartpole problem.
This is a problem in which a cart is carrying a pole and must balance it. As it chooses
an action to take in the way it moves, it measures how much of a reward it receives.
Then it uses that reward to update the value table and continue making decisions.
"""

env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

"""
Build a single-layer hidden neural network.
"""

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation("relu"))
model.add(Dense(nb_actions))
model.add(Activation("linear"))
print(model.summary())

"""
Configure and compile the agent. We use epsilon greedy that either takes a random
action with probability epsilon or takes the current best action with probability
1 - epsilon. SequentialMemory means that we make observations, take actions,
receive a reward, and then choose the next state. We use mean absolute error (mae)
for our metrics for our loss metrics.
"""

policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
      target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=["mae"])

"""
We visualize the training here for show, but this slows down training quite a lot. 
"""

dqn.fit(env, nb_steps=5000, visualize=True, verbose=2)
