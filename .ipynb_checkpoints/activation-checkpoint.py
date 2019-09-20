import numpy as np

class Identity:
    def __init__(self):
        self.name = 'identity'

    def __call__(self, x):
        return x

    def dev(self, x):
        return np.ones_like(x)

class ReLU:
    def __init__(self):
        self.name = 'relu'

    def __call__(self, x):
        return (x>0)*x

    def dev(self, x):
        return (x>=0)*1.0
    
class Softplus: #  approximation RELU
    def __init__(self):
        self.name = 'softplus'

    def __call__(self, x):
        return np.log(1.0 + np.exp(x))

    def dev(self, x):
        return 1.0 / (1.0 + np.exp(-x) ) 

class Sigmoid:
    def __init__(self):
        self.name = 'sigmoid'

    def __call__(self, x):
        return 1/(1+np.exp(-x))

    def dev(self, x):
        return (1/(1+np.exp(-x)))*(1-1/(1+np.exp(-x)))

activation_fns_d = {
    'relu': ReLU() ,
    'softplus': Softplus() ,
    'sigmoid': Sigmoid() ,
    'identity': Identity(),

}