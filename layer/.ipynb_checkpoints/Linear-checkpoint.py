import numpy as np
from .layer import Layer
from initialization import *
from activation import *

class Linear(Layer):
    
    """
    <foward>
    X: (n_batch, n_features), dtype: float 
    W: (n_features, n_features_out), dtype: float (tunable)
    B: (n_features_out, ), dtype: float (tunable)
    Z: (n_batch, n_features_out), dtype: float , WX+B
    foward_output: (n_batch, n_features_out), dtype: float
    <backward>
    G_z, G_out: (n_batch, n_features_out), must the same with Z
    G_in:  (n_batch, n_features), must the same with X
    G_w: (n_features, n_features_out), must the same with W
    G_b: (n_features_out, ), must the same with B
    """ 
    
    def __init__(self, dim_in, dim_out, init_method='random_normal', act='relu'):
        super(Linear, self).__init__()
        self.X = None
        self.Z = None
        self.G_w = None
        self.G_b = None
        self.W = initial_methods_d[init_method](size=(dim_in,dim_out))
        self.B = initial_methods_d[init_method](size=(dim_out,))
        self.act = activation_fns_d[act]
    
    def forward(self, X):
        self.X = X
        self.Z = np.dot(X, self.W) + self.B
        return self.act(self.Z)
    
    def backward(self, G_out):
        assert (not self.X is None) and (not self.Z is None) 
        deactivate = self.act.dev(self.Z)
        G_z = G_out*deactivate
        G_in = np.dot(G_z, self.W.T )
        self.G_w = np.dot(self.X.T, G_z )
        self.G_b = np.sum(G_z, 0)
        return G_in
    
    def step(self, optimizer):
        assert (not self.G_w is None) and (not self.G_b is None) 
        self.W = optimizer(self.W, self.G_w)
        self.B = optimizer(self.B, self.G_b)
        
    def zero_gradient(self):
        self.G_w = None
        self.G_b = None
    