import numpy as np
from Data_Pipeline import *
N_Classes = 10
maxnp = np.log(np.finfo(np.float32).max)
def relu(z:np.ndarray):
    return np.maximum(0,z)#Element wise Maximum Operation
def softmax(z:np.ndarray):
    exps = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)
def one_hot_encoder(train_labels:np.ndarray,N_Classes = 10):
    arr = np.eye(N_Classes)
    encoded = arr[train_labels]
    return encoded
def Cross_EntropyLoss(Afinal:np.ndarray,encoded:np.ndarray):
    Y_c = np.sum(encoded*Afinal,axis = 1)
    Y_c = np.clip(Y_c, 1e-15, 1.0)
    L = -np.log(Y_c)
    return(np.mean(L))



