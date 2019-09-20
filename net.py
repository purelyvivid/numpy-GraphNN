import numpy as np
from layer.Linear import Linear
from layer.GCNconv import GCNconv
from activation import *
from initialization import *
from optimizer import *
from loss import *

class Net():
    
    def __init__(self, 
                 dims=[10,3,1],
                 layer_type = 'linear',
                 init_method='random_normal', 
                 act='relu', 
                 opfn='identity',
                ):
        
        self.dims = dims
        self.layer_type = layer_type
        self.layer = GCNconv if layer_type=='gcn' else Linear
        self.layers = []
        self.loss_ = 'mse'
        print("---<Model Summary>---")
        for i, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            act_ = opfn if i==len(dims)-2 else act
            layer = self.layer(d_in, d_out, init_method=init_method, act=act_)
            print('layer', i+1, ':   dim =', (d_in, d_out), ', act =', act_)
            self.layers.append(layer) 
        print("output_fn: ", opfn)
        print("---------------------")
        
    def forward(self, X, **kwargs):
        for layer in self.layers:
            X = layer(X, **kwargs)
        return X
    
    def __call__(self, X, **kwargs):
        return self.forward(X, **kwargs)    

    def backward(self, G_out):
        for layer in reversed(self.layers):
            G_out = layer.backward(G_out)
        return G_out
        
    def step(self,  opt):
        for layer in reversed(self.layers):
            layer.step(opt)
            
    def zero_gradient(self):
        for layer in self.layers:
            layer.zero_gradient() 
            
    def fit(self, X, y, 
            loss_='mse', 
            lr=0.01, 
            opt='sgd', 
            epochs=10,
            print_freq=1,
            **kwargs
           ):
        print("")
        print("------<Train>------")
        print("loss_fn: ", loss_, ", optimizer: ", opt, ", start_lr: ", lr)
        print("")
        self.loss_ = loss_
        loss_fn = loss_fns_d[loss_]
        optimizer = optimizors_d[opt](lr=lr)
        for i in range(epochs):
            x_output = self.forward(X, **kwargs)
            loss = loss_fn(x_output, y)
            if i==0 or i%print_freq==0 or i==epochs-1: print("loss: ",loss)
            self.zero_gradient()
            grad = loss_fn.backward()
            _ = self.backward(grad)
            self.step(optimizer)
        print("---------------------")
        
    def score(self, X, y, loss_=None, **kwargs):
        if loss_ is None:
            loss_ = self.loss_
        x_output = self.forward(X, **kwargs)
        loss_fn = loss_fns_d[loss_]
        return loss_fn.score(x_output, y)
        
        