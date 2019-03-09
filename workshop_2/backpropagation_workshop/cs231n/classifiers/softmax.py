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
    batch_size = X.shape[0]
    
    for i in range(batch_size):
        # (1, D) * (D, C) = (, C)
        f = np.dot(X[i], W)
#        print(f.shape)
        
        # scalar
        b = np.max(f)
        
        # scalar
        f_sum = np.sum(np.exp(f - b))
        
        # scalar
        loss += -f[y[i]] + b + np.log(f_sum)
        
        # (1, C)
        y_ohe = np.zeros(W.shape[1])
        y_ohe[y[i]] = 1
        
        # (C, ) - (C, ) = (C, )
        soft_grad = f - y_ohe
#        
#        # (D, 1) * ((1, C) - (1, C)) = (D, C)
        dW += np.dot(X[i].reshape(-1, 1), soft_grad.reshape(1, -1))
        
    loss = (loss / batch_size) + reg * np.sum(W**2)
#    dW /= batch_size
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
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
