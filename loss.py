import numpy as np

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, 1, keepdims=True)
    return softmax_x    

def polyfit(x, y):
    correlation = np.corrcoef(x, y)[0,1]
    return correlation**2

class MSELoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def __call__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean(np.sum(np.square(y_pred - y_true), 1))

    def backward(self):
        dy_pred = 2 * (self.y_pred - self.y_true) / self.y_pred.size
        return dy_pred
    
    def score(self, y_pred, y_true): # R-sq
        return polyfit(y_pred.flatten(), y_true.flatten())
    
class BCEWithLogitsLoss:
    def __init__(self):
        self.y_pred_logits = None
        self.y_true = None

    def __call__(self, x_output, y_true):
        self.y_pred_logits = softmax(x_output)
        self.y_true = y_true
        return -np.sum(self.y_true*np.log(self.y_pred_logits))/len(self.y_true)

    def backward(self):
        dy_pred = self.y_pred_logits - self.y_true
        return dy_pred  
    
    def score(self, x_output, y_true): 
        y_pred_logits = softmax(x_output)
        y_pred_labels = np.argmax(y_pred_logits, 1)
        y_true_labels = np.argmax(y_true, 1)
        return sum(y_pred_labels==y_true_labels)/len(y_pred_labels)
    
loss_fns_d = {
    'mse': MSELoss() ,
    'bce_logits': BCEWithLogitsLoss() ,
}