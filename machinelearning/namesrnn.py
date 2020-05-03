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

token_to_id = {token:i for i,token in enumerate(tokens)}

def to_matrix(names,max_len=None,pad=0,dtype="int32"):
    """
    Casts a list of names into rnn-digestable matrix.
    """
    
    max_len = max_len or max(map(len,names))
    names_ix = np.zeros([len(names),max_len],dtype) + pad

    for i in range(len(names)):
        name_ix = list(map(token_to_id.get,names[i]))
        names_ix[i,:len(name_ix)] = name_ix

    return names_ix.T
