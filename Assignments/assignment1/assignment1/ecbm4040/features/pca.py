import time
import numpy as np

def pca_naive(X, K):
    """
    PCA -- naive version

    Inputs:
    - X: (float) A numpy array of shape (N, D) where N is the number of samples,
         D is the number of features
    - K: (int) indicates the number of features you are going to keep after
         dimensionality reduction

    Returns a tuple of:
    - P: (float) A numpy array of shape (K, N), representing the top K
         principal components
    - T: (float) A numpy vector of length K, showing the score of each
         component vector
    """

    ###############################################
    #TODO: Implement PCA by extracting eigenvector#
    ###############################################
    mean_vector = np.mean(X, axis = 0)
    N = X.shape[0]
    D = X.shape[1]
    cov = (np.matrix(X).T.dot(X))/(N-1)
    eigen_val, eigen_vec = np.linalg.eig(cov)
    
    idx = eigen_val.argsort()[::-1]   
    eigen_val = eigen_val[idx]
    eigen_vec = eigen_vec[:,idx]
    
    T=eigen_val[0:K]
    P_temp = eigen_vec[:,0:K]
    
    eigvectors = []
    for i in range(K):
        eigvectors.append(P_temp[:,i])
        
    P = np.matrix(P_temp).T
    
    
    
    ###############################################
    #              End of your code               #
    ###############################################
    
    return (P, T)
