import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    scores = X[i].dot(W)    # (C,)
    scores -= np.max(scores)    # To prevent numeric instability (see http://cs231n.github.io/linear-classify/)
    Nr = np.exp(scores[y[i]])
    Dr = 0
    for j in xrange(num_classes):
      Dr += np.exp(scores[j])
    for j in xrange(num_classes):
      # if j == y[i]:
      #    dW[:, j] += (np.exp(scores[j])/Dr - 1) * X[i]
      # if j != y[i]:
      #    dW[:, j] += (np.exp(scores[j])/Dr) * X[i]
      dW[:, j] += (np.exp(scores[j])/Dr - (j == y[i])) * X[i]    # (D,)
    loss += -np.log(Nr/Dr)
  loss /= num_train  
  dW /= num_train
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W)    # (N, C)
  scores -= np.max(scores, axis=1, keepdims=True)    # To prevent numeric instability (see http://cs231n.github.io/linear-classify/)
  exp_scores = np.exp(scores)    # (N, C)
  Nr = exp_scores[np.arange(num_train), y]    # (N,)
  Dr = np.sum(exp_scores, axis=1)    # (N,)
  frac = Nr/Dr    # (N,)
  log_frac = - np.log(frac)    # (N,)
  loss = np.sum(log_frac) / num_train
  loss += reg * np.sum(W * W)
  
  dlog_frac = (1.0/num_train) * np.ones_like(log_frac)    # (N,)
  dfrac = dlog_frac * (-1.0/frac)    # (N,)
  dDr = (dfrac * (-Nr/Dr**2)).reshape(num_train, 1)    # (N, 1)
  dNr = (dfrac * (1.0/Dr)).reshape(num_train, 1)    # (N, 1)

  dNr_by_dexp_scores = np.zeros_like(exp_scores)    # (N, C)
  dNr_by_dexp_scores[np.arange(num_train), y] = 1    # (N, C)
  dexp_scores = dNr * dNr_by_dexp_scores    # (N, C)
  
  dDr_by_dexp_scores = np.ones_like(exp_scores)    # (N, C)
  dexp_scores += dDr * dDr_by_dexp_scores    # (N, C)
   
  dscores = dexp_scores * exp_scores    # (N, C)
  dW = (X.T).dot(dscores)    # (D, C)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

