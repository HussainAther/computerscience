import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from keras import losses
from keras.layers import concatenate,Dense,Embedding
from random import sample

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

rnn_num_units = 64
embedding_size = 16

embed_x = Embedding(n_tokens,embedding_size) # an embedding layer that converts character ids into embeddings

# a dense layer that maps input and previous state to new hidden state, [x_t,h_t]->h_t+1
get_h_next = Dense(rnn_num_units, activation="tanh")

# a dense layer that maps current hidden state to probabilities of characters [h_t+1]->P(x_t+1|h_t+1)
get_probas = Dense(n_tokens, activation="softmax")

def rnn_one_step(x_t, h_t):
    """
    Recurrent neural network step that produces next state and output
    given prev input and previous state.
    """
    # Convert character id into embedding.
    x_t_emb = embed_x(tf.reshape(x_t,[-1,1]))[:,0]
    
    # Concatenate x embedding and previous h state.
    x_and_h = concatenate([x_t_emb, h_t])
    #print(x_and_h.get_shape().as_list())
    
    # Compute next state given x_and_h.
    h_next = get_h_next(x_and_h)
    #print(h_next.get_shape().as_list())
    
    # Get probabilities for language model P(x_next|h_next).
    output_probas = get_probas(h_next)
    
    return output_probas,h_next

input_sequence = tf.placeholder("int32",(MAX_LENGTH,None))
batch_size = tf.shape(input_sequence)[1]
predicted_probas = []
h_prev = tf.zeros([batch_size,rnn_num_units]) #initial hidden state

for t in range(MAX_LENGTH):
    x_t = input_sequence[t]
    probas_next,h_next = rnn_one_step(x_t,h_prev)
    h_prev = h_next
    predicted_probas.append(probas_next)
    
predicted_probas = tf.stack(predicted_probas)
predictions_matrix = tf.reshape(predicted_probas[:-1],[-1,len(tokens)])
answers_matrix = tf.one_hot(tf.reshape(input_sequence[1:],[-1]), n_tokens)
loss = losses.categorical_crossentropy(answers_matrix, predictions_matrix)
optimize = tf.train.AdamOptimizer().minimize(loss)

# training loop
s = keras.backend.get_session()
s.run(tf.global_variables_initializer())
history = []

# sampling
x_t = tf.placeholder("int32",(1,))
h_t = tf.Variable(np.zeros([1,rnn_num_units],"float32"))
next_probs,next_h = rnn_one_step(x_t,h_t)

def generate_sample(seed_phrase=" ",max_length=MAX_LENGTH):
    """
    The function generates text given a phrase of length at least SEQ_LENGTH.
        
    parameters:
        The phrase is set using the variable seed_phrase
        The optional input "N" is used to set the number of characters of text to predict.     
    """
    x_sequence = [token_to_id[token] for token in seed_phrase]
    s.run(tf.assign(h_t,h_t.initial_value))
    for ix in x_sequence[:-1]:
         s.run(tf.assign(h_t,next_h),{x_t:[ix]})
    
    # Generate.
    for _ in range(max_length-len(seed_phrase)):
        x_probs,_ = s.run([next_probs,tf.assign(h_t,next_h)],{x_t:[x_sequence[-1]]})
        x_sequence.append(np.random.choice(n_tokens,p=x_probs[0]))
        
    return "".join([tokens[ix] for ix in x_sequence])
