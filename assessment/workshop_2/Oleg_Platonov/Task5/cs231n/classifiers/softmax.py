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
    def softmax(scores, i):
        return np.exp(scores[i]) / np.sum(np.exp(scores))

    num_train = X.shape[0]
    num_classes = W.shape[1]

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in range(num_train):
        scores = X[i].dot(W)
        loss -= np.log(softmax(scores, y[i]))
        for j in range(num_classes):
            dW[:,j] -= ((y[i]==j) - softmax(scores, j)) * X[i]

    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
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
    def softmax_matrix(scores):
        exp = np.exp(scores)
        return exp / np.sum(exp, axis=1, keepdims=True)

    num_train = X.shape[0]
    num_classes = W.shape[1]

    one_hot_y = np.eye(num_classes)[y]

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores = X @ W
    softmax = softmax_matrix(scores)

    loss -= np.sum(np.log(softmax[np.where(one_hot_y)])) / num_train
    dW -= X.T @ (one_hot_y - softmax) / num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg*2*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
