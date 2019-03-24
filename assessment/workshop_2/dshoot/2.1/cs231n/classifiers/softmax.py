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
    D = W.shape[0]
    C = W.shape[1]
    N = X.shape[0]
    for s in range(N):
        xx = X[s,:]
        scores = np.dot(xx, W)
        scores -= np.max(scores) # for numerical stability
        yy = y[s]
        loss -= scores[yy]
        norm_factor = 0.0
        for i in range(C):
            norm_factor += np.exp(scores[i])
        loss += np.log(norm_factor)
        # gradients
        for i in range(C):
            softmax_i = np.exp(scores[i]) / norm_factor
            if i == yy:
                softmax_i -= 1.0
            for f in range(D):
                dW[f,i] += softmax_i*xx[f]
    # normalization
    loss /= N
    dW /= N
    # regularization
    loss += reg*np.sum(W**2)
    dW += reg*2*W
    
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
    D = W.shape[0]
    C = W.shape[1]
    N = X.shape[0]
    
    scores = np.dot(X, W)
    scores -= np.reshape(np.max(scores, axis=1), [N,1]) # for numerical stability
    loss -= np.sum(scores[range(N),y])
    score_exps = np.exp(scores)
    norm_factor = np.sum(score_exps, axis=1)
    loss += np.sum(np.log(norm_factor))
    # gradients
    softmax = score_exps / np.reshape(norm_factor, [N,1]) # N*C
    softmax[range(N),y] -= 1.0
    X_C = np.transpose(np.broadcast_to(X, [C]+list(X.shape))) # D*N*C
    dW = np.sum(np.multiply(X_C, softmax), axis=1)
    # normalization
    loss /= N
    dW /= N
    # regularization
    loss += reg*np.sum(W**2)
    dW += reg*2*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
