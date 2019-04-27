import numpy as np
import matplotlib.pyplot as plt

"""
Implementation of a two-layer neural network. 
"""

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer. In other words, the network has the following architecture:
  input is a fully connected layer. ReLU is a fully connected layer softmax.
  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params["W1"] = std * np.random.randn(input_size, hidden_size)
    self.params["b1"] = std * np.random.randn(hidden_size)  + 0.5  # np.zeros(hidden_size)
    self.params["W2"] = std * np.random.randn(hidden_size, output_size)
    self.params["b2"] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network. X is an input data of shape (N, D). Each X[i] is a training sample.
    y is a vector of training labels. y[i] is the label for X[i], and each y[i] is
    an integer in the range 0 <= y[i] < C. This parameter is optional; if it
    is not passed then we only return scores, and if it is passed then we
    instead return the loss and gradients.
    reg is regularization strength.
    Returns, if y is None,  a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i]. If y is not None, instead return a tuple of
    loss: Loss (data loss and regularization loss) for this batch of training samples.
    grads: Dictionary mapping parameter names to gradients of those parameters
    with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params["W1"], self.params["b1"]
    W2, b2 = self.params["W2"], self.params["b2"]
    N, D = X.shape

    """
    Compute the forward pass
    """
    scores = None
    z = np.dot(X, W1) + b1  # (N, num_hidden)
    h = np.maximum(z, 0)    # ReLU
    scores = np.dot(h , W2) + b2
    # If the targets are not given then jump out, we"re done
    if y is None:
      return scores

    """
    Compute the loss
    """
    loss = 0.0
    
    """
    Compute softmax probabilities
    """
    out = np.exp(scores)      # (N, C)
    out /= np.sum(out, axis=1).reshape(N, 1)
    
    """
    Compute softmax loss
    """
    loss -= np.sum(np.log(out[np.arange(N), y]))
    loss /= N
    loss += 0.5 * reg * (np.sum(W1**2) + np.sum(W2**2))
    
    """
    Backward pass: compute gradients
    """
    grads = {}
    
    """
    Backpropagation
    """
    dout = np.copy(out)  # (N, C)
    dout[np.arange(N), y] -= 1
    dh = np.dot(dout, W2.T)
    dz = np.dot(dout, W2.T) * (z > 0)  # (N, H)
    
    """
    Compute gradient for parameters
    """
    grads["W2"] = np.dot(h.T, dout) / N # (H, C)
    grads["b2"] = np.sum(dout, axis=0) / N # (C,)
    grads["W1"] = np.dot(X.T, dz) / N # (D, H)
    grads["b1"] = np.sum(dz, axis=0) / N # (H,)
    
    """
    Add reg term
    """
    grads["W2"] += reg * W2
    grads["W1"] += reg * W1
    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.
    X is a numpy array of shape (N, D) giving training data.
    y is a numpy array f shape (N,) giving training labels; y[i] = c means that
    X[i] has label c, where 0 <= c < C. X_val is a numpy array of shape (N_val, D)
    giving validation data. y_val is a numpy array of shape (N_val,) giving validation labels.
    learning_rate is a scalar giving learning rate for optimization.
    learning_rate_decay is a scalar giving factor used to decay the learning rate after each epoch.
    reg is a scalar giving regularization strength.
    num_iters is the number of steps to take when optimizing.
    batch_size is the number of training examples to use per step.
    verbose, if true, prints progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None
      random_idxs = np.random.choice(num_train, batch_size)
      X_batch = X[random_idxs]
      y_batch = y[random_idxs]
      """
      Compute loss and gradients using the current minibatch
      """
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)
      self.params["W2"] -= learning_rate * grads["W2"]
      self.params["b2"] -= learning_rate * grads["b2"]
      self.params["W1"] -= learning_rate * grads["W1"]
      self.params["b1"] -= learning_rate * grads["b1"]
      if verbose and it % 100 == 0:
        print("iteration %d / %d: loss %f" % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      "loss_history": loss_history,
      "train_acc_history": train_acc_history,
      "val_acc_history": val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.
    X is  numpy array of shape (N, D) giving N D-dimensional data points to
    classify. Return y_pred, a numpy array of shape (N,) giving predicted labels for each of
    the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
    to have class c, where 0 <= c < C.
    """
    y_pred = None
    params = self.params
    z = np.dot(X, params["W1"]) + params["b1"]
    h = np.maximum(z, 0)
    out = np.dot(h, params["W2"]) + params["b2"]
    y_pred = np.argmax(out, axis=1)
    return y_pred
