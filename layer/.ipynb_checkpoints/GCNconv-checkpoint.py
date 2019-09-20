import numpy as np
from .layer import Layer
from .Linear import Linear
from initialization import *
from activation import *

class GCNconv(Layer):
    """
    <foward>
    X: (n_node, n_features), dtype: float 
    A: (n_node, n_node), dtype:  0-1 matrix
    foward_output: (n_node, n_features_out), dtype: float
    <backward>
    G_out, G_temp: (n_node, n_features_out)
    
    """
    
    def __init__(self, dim_in, dim_out=None, init_method='random_normal', act='identity'):
        super(GCNconv, self).__init__()
        self.N = None
        if dim_out is None: dim_out = dim_in
        self.fc = Linear(dim_in, dim_out, init_method=init_method, act=act)
        
    def norm(self, A):
        n = A.shape[0]
        I = np.diag(np.ones((n)))
        A_hat = A + I
        D_hat = np.diag(np.sum(A_hat,axis=0)) # axis=0: degree_out
        with np.errstate(divide='ignore'):
            D_hat_inv_sqrt = D_hat**(-0.5)
            D_hat_inv_sqrt[D_hat_inv_sqrt == float('inf')] = 0 # prevent from exploding
        return D_hat_inv_sqrt.dot(A).dot(D_hat_inv_sqrt) # D^(-0.5)*A*D^(-0.5)
    
    def forward(self, X, A):
        if (self.N is None) or (not A is None):
            self.N = self.norm(A) # catched for the first time, then fixed
        return self.fc(np.dot(self.N, X))
    
    def backward(self, G_out):
        G_temp = self.fc.backward(G_out)
        G_in = np.dot(self.N.T, G_temp)
        return G_in
    
    def step(self, optimizer):  
        self.fc.step(optimizer)
        
    def zero_gradient(self):
        self.fc.zero_gradient()
                 