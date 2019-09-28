import numpy as np

class RandomNormal():
    def __init__(self):
        self.name = 'random_normal'

    def __call__(self, size):
        return np.random.normal(0.0, 0.01, size=size)  
    
class RandomUniform():
    def __init__(self):
        self.name = 'random_uniform'

    def __call__(self, size):
        return np.random.normal(-0.001, 0.001, size=size) 

initial_methods_d = {
    'random_normal': RandomNormal(),
    'random_uniform': RandomUniform(),
}