import numpy as np

class SGD:
    def __init__(self, lr=0.001):
        self.name = 'sgd'
        self.lr = lr

    def __call__(self, param, gradient):
        return self.update(param, gradient)
    
    def update(self, param, gradient):
        return param - self.lr*gradient
    
optimizors_d = {
    'sgd': SGD ,
    #'adam': Adam ,
} 