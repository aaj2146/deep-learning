import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wst W, an array of same shape as W
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
    num_samples = X.shape[0]
    
    for i in range(num_samples):
        score_row = X[i].dot(W)
        score_row = score_row - np.max(score_row) #Normalizing the matrix
        probabilities = (np.exp(score_row))/(np.sum(np.exp(score_row)))
        probability = (np.exp(score_row[y[i]]))/(np.sum(np.exp(score_row)))
        
        loss += -np.log(probabilities[y[i]])
        
        for j in range(num_classes):
            dW[:,j] += X[i,:] * probabilities[j]
          
        
     
    
    loss = loss/num_samples
    
    
    dW /= num_samples
    
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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
    N = X.shape[0]
    C = W.shape[1]

    score_matrix = X.dot(W)
    score_matrix -= np.matrix(np.max(score_matrix, axis=1)).T
    
    exp_score_matrix = np.exp(score_matrix)
    
    exp_correct_scores = exp_score_matrix[np.arange(N), y]
    
    #denominator = np.sum(exp_score_matrix, axis=1)
    
    probabilities = exp_score_matrix/np.sum(np.exp(score_matrix),axis=1,keepdims=True)
    
    correct_class_prob = probabilities[np.arange(N), y]
    
    loss = np.sum(-np.log(correct_class_prob)) / N
    
    loss += reg * np.sum(W*W) 
    
    probabilities[range(N),y] -= 1
    dW = X.T.dot(probabilities) / N
    
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
