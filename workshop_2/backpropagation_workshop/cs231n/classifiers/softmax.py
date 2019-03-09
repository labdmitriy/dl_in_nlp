import numpy as np
from random import shuffle

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
    num_classes = W.shape[1]
    
    for i in range(num_train):
        # (C, )
        scores = np.zeros(num_classes)
        
#        for j in range(num_classes):
#            # (1, D) * (D, 1) = scalar
#            scores[j] = np.dot(X[i, :], W[:, j])
        
        # (C, )
        scores = np.dot(X[i], W)
        
        # (C, )
        scores -= np.max(scores)
        
        # (C, )
        y_ohe = np.zeros(num_classes)
        y_ohe[y[i]] = 1
        
        # (C, )
        scores_exp = np.exp(scores)
        
        # scalar
        loss += -np.log(scores_exp[y[i]] / np.sum(scores_exp))
        
        # (D, 1) * ((1, C) / scalar - (1, C)) = (D, C)
        dW += X[i].reshape(-1, 1) * (scores_exp / np.sum(scores_exp) - y_ohe)
    
    # scalar
    loss /= num_train
    
    # (D, C)
    dW /= num_train
    
    # scalar 
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
    
    # (N, C)
    scores = np.dot(X, W)
    
    # (N, C)
    scores -= np.max(scores, axis=1, keepdims=True)
    
    # (N, C)
    y_ohe = np.zeros_like(scores)
    y_ohe[np.arange(num_train), y] = 1
    
    # (N, C)
    scores_exp = np.exp(scores)
    
    # mean((N, ) / (N, )) = scalar
    loss = np.mean(-np.log(scores_exp[np.arange(num_train), y] / np.sum(scores_exp, axis=1)))
    loss += reg * np.sum(W * W)
    
    # (D, N) * ((N, C) / scalar - (N, C)) = (D, C)
    dW = np.dot(X.T, scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
               - y_ohe) / num_train
    dW += 2 * reg * W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
