import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

"""
Recurrent neural networks with tensorflow
"""

def to_matrix(names, max_len=None, pad=0, dtype="int32"):
    """
    Cast a list of names into rnn-digestable matrix.
    """
    
    max_len = max_len or max(map(len,names))
    names_ix = np.zeros([len(names),max_len],dtype) + pad

    for i in range(len(names)):
        name_ix = list(map(token_to_id.get,names[i]))
        names_ix[i,:len(name_ix)] = name_ix

    return names_ix.T
