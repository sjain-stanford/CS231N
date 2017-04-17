import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
        # Gradient (see http://cs231n.github.io/optimization-1/#analytic)
        # w.r.t. weights of incorrect class
        dW[:,j] += X[i]
        # w.r.t. weights of correct class
        dW[:,y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += (2 * reg * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  """ Easy implementation
  num_train = X.shape[0]
  scores = X.dot(W)  # NxC
  # Advanced indexing: http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
  correct_class_score = scores[np.arange(num_train), y]   # 1xN  
  margins = np.maximum(0, scores.T - correct_class_score + 1)   # CxN
  margins = margins.T  # NxC  
  margins[np.arange(num_train), y] = 0   # Setting the j=y_i terms zero since sum doesn't include them
  data_loss = np.sum(margins)/num_train
  reg_loss = reg * np.sum(W * W)   # Regularization loss
  loss = data_loss + reg_loss
  """
  """ Detailed forward-pass """
  num_train = X.shape[0]
  scores = X.dot(W)    # (N, C)
  correct_class_scores = scores[np.arange(num_train), y].reshape(num_train, -1)   # (N, 1)
  margins0 = (scores - correct_class_scores + 1)    # (N, C)
  margins1 = np.maximum(0, margins0)    # (N, C)
  margins2 = margins1    # (N, C)   Copy to ease the back-prop later
  margins2[np.arange(num_train), y] = 0    # (N, C)   # Setting the j=y_i terms zero since sum doesn't include them
  data_loss = np.sum(margins2) / num_train    # ()
  reg_loss = reg * np.sum(W * W)    # ()
  loss = data_loss + reg_loss
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  ###      Notation           upstream grad     local grad
  ### dvar = d(loss)/d(var) = d(loss)/d(out) * d(out)/d(var)
  """  Easy implementation
  dscores = (margins>0)*1  # NxC
  dscores[np.arange(num_train), y] = -np.sum(dscores, axis=1)   # NxC
  dW = (X.T).dot(dscores)   # DxC
  dW /= num_train
  dW += 2 * reg * W   # DxC
  """
  """ Detailed back-prop """
  # dmargins2 = d(loss)/d(margins2)
  dmargins2 = (1.0/num_train) * np.ones_like(margins2)    # (N, C)

  # dmargins1 = d(loss)/d(margins1) = d(loss)/d(margins2) * d(margins2)/d(margins1)
  dmargins1 = dmargins2    # (N, C)
  dmargins1[np.arange(num_train), y] = 0    # (N, C)

  # dmargins0 = d(loss)/d(margins0) = d(loss)/d(margins1) * d(margins1)/d(margins0)
  dmargins0 = dmargins1 * (margins0 > 0)    # (N, C)

  # dscores = d(loss)/d(scores) = d(loss)/d(margins0) * d(margins0)/d(scores)
  dscores = dmargins0    # (N, C)
  
  # dcorr_scores = d(loss)/d(corr_scores) = d(loss)/d(margins0) * d(margins0)/d(corr_scores)
  #  Dimensional analysis     (N, 1)      =      (N, C)       .dot        (C, 1)                      
  dmargins0_by_dcorr_scores = -1.0 * np.ones([np.shape(margins0)[1], 1])
  dcorr_scores = dmargins0.dot(dmargins0_by_dcorr_scores)    # (N, 1)
  
  # dscores = d(loss)/d(scores) = d(loss)/d(corr_scores) * d(corr_scores)/d(scores)
  #  Dimensional analysis (N, C) =       (N, 1)          *        (N, C)    (element-wise multiplication)
  dcorr_scores_by_dscores = np.zeros_like(scores)
  dcorr_scores_by_dscores[np.arange(num_train), y] = 1
  dscores += dcorr_scores * dcorr_scores_by_dscores    # (N, C)    (element-wise multiplication)
  
  # dW = d(loss)/d(W) = d(scores)/d(W) * d(loss)/d(scores)
  #        (D, C)     =     (D, N)   .dot    (N, C)
  # scores = X.dot(W) ---> d(scores)/d(W) = X.T (transposed)    # (D, N)
  dW = (X.T).dot(dscores)    # (D, C)
  dW += 2 * reg * W    # (D, C)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
