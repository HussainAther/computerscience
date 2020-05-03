import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

"""
Generate names with recurrent neural networks.
"""

start_token = " "

with open("names") as f:
    names = f.read()[:-1].split("\n")
    names = [start_token+name for name in names]

MAX_LENGTH = max(map(len,names))
print("max length =", MAX_LENGTH)

plt.title("Sequence length distribution")
plt.hist(list(map(len,names)),bins=25)

# Character tokens
# All unique characters go here
tokens = set()
for name in names:
    tokens = tokens.union(set(name))

tokens = list(tokens)

n_tokens = len(tokens)
print("n_tokens = ",n_tokens)

assert 50 < n_tokens < 60

