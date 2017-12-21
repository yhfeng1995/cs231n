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
  num_train = X.shape[0]
  num_feature = X.shape[1]
  num_class = W.shape[1]
  s = X.dot(W)
  
  for i in range(num_train):
        sum_exp_s = np.sum(np.exp(s[i]))
        exp_s_yi = np.exp(s[i][y[i]])
        
        # loss: Li
        p_i = exp_s_yi / sum_exp_s
        loss += -np.log(p_i)
        
        # gradient: d(Li)/d(wm), m!=yi
        # 1. gradient: d(Li)/d(sm)
        dsm = 1/p_i * exp_s_yi/np.power(sum_exp_s, 2) * np.exp(s[i])
        # 2. gradient: d(Li)/d(wm)
        dwm = X[i].reshape(num_feature, 1).dot(dsm.reshape(1, num_class))
        
        # gradient: d(Li)/d(wyi)
        dsyi_extra = -exp_s_yi/(p_i*sum_exp_s) 
        dyi_extra_v = dsyi_extra * X[i]
        dyi_extra = np.zeros(W.shape)
        dyi_extra[:, y[i]] = dyi_extra_v
        
        dwm += dyi_extra
        dW +=dwm
  # end for

  loss = loss/num_train + reg*np.sum(np.power(W, 2))
  dW = dW / num_train + reg*2*W
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
  num_feature = X.shape[1]
  num_class = W.shape[1]
  
  s = X.dot(W)
  exp_s_yi = np.exp(s[np.arange(num_train), y])
  sum_exp_s = np.sum(np.exp(s), 1)
  # loss
  loss = np.sum(-np.log(exp_s_yi/sum_exp_s)) / num_train + reg*np.sum(np.power(W, 2))
  
  # gradient
  # 1. d(L)/dWm, m!=y
  p = exp_s_yi / sum_exp_s
  # print p.shape, exp_s_yi.shape, np.power(sum_exp_s, 2).shape
  ds = (1/sum_exp_s).reshape(num_train, 1) * np.exp(s)
  dwm = X.T.dot(ds)
  
  # 2. d(L)/dWyi
  ds_y_extra_m = np.zeros(s.shape)
  ds_y_extra_m[np.arange(num_train), y] = -1
  dw_y_extra = X.T.dot(ds_y_extra_m)
  
  # print dwm.shape, dw_y_extra.shape
  dW = (dwm + dw_y_extra)/num_train + 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

