class Layer:
    
    def __init__(self, **kwargs):
        pass
    
    def forward(self, **kwargs):
        pass
    
    def backward(self, **kwargs):
        pass
    
    def step(self, **kwargs):
        pass
        
    def zero_gradient(self, **kwargs):
        pass
    
    def __call__(self, X, **kwargs):
        return self.forward(X, **kwargs) 