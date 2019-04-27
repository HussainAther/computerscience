import numpy as np
from random import shuffle

"""
Linear support vector machine (SVM svm) for computer vision.
"""

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    W is a numpy array of shape (D, C) containing weights.
    X is a numpy array of shape (N, D) containing a minibatch of data.
    y is a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
    reg is (float) regularization strength
    Return a tuple of loss as single float and agradient with respect to weights W;
    an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]  # this is only added
                dW[:, y[i]] -= X[i]
  
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.
    Inputs and outputs are the same as svm_loss_naive.
    """
    C = W.shape[1]
    N = X.shape[0]
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    scores = np.dot(X, W) # (N, C)
    correct_scores = scores[np.arange(N), y] # (N, )
    margins = np.maximum(scores - correct_scores.reshape(N, 1) + 1.0, 0)  # (N, C)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    loss += 0.5 * reg * np.sum(W * W)
    dscores = np.zeros_like(scores) # (N, C)
    dscores[margins > 0] = 1
    dscores[np.arange(N), y] -= np.sum(dscores, axis=1) #  (N, 1) = (N, 1)
    dW = np.dot(X.T, dscores)
    dW /= N
    dW += reg * W
    return loss, dW
