from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    for num_id in range(num_train):
        f = X[num_id] @ W
        f -= np.max(f)
        f_exp = np.exp(f)
        f_sum = np.sum(f_exp)
        for class_id in range(num_classes):
            prob = f_exp[class_id] / f_sum
            if class_id == y[num_id]:
                dW[:, class_id] +=  (-1 + prob) * X[num_id]
                continue
            dW[:, class_id] +=  prob * X[num_id]
        loss += - np.log(f_exp[y[num_id]] / f_sum)

    loss /= num_train
    loss += 1 / 2 * reg * np.sum(W * W)

    dW /= num_train
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = X.shape[0]

    f = X @ W
    f -= np.max(f, axis=1).reshape(num_train, -1)
    f_exp = np.exp(f)
    prob = f_exp / np.sum(f_exp, axis=1).reshape(-1, 1)
    
    loss = np.sum(- np.log(prob[range(num_train),
                                y[range(num_train)]]))
    
    prob[range(num_train), y[range(num_train)]] -= 1
    dW += X.T @ prob
   
    loss /= num_train
    loss += 1 / 2 * reg * np.sum(W * W)

    dW /= num_train
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
