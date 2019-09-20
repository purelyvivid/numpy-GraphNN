import numpy as np

def to_categorical(y_, n_class):
    y = np.zeros((y_.shape[0], n_class)) #init
    for i, j in enumerate(y_):
        y[i, j] = 1
    return y