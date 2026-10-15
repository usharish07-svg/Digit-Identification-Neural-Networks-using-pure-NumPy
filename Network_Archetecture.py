import numpy as np
from Data_Pipeline  import *
from Special_Functions import *
# 1 Input Layer(784), 2 Hidden Layers(128), 1 Output Layer(10)

#Weight Creation
def parameter_intitialisation():
    W_i1 = np.random.randn(784,128)
    W1 = W_i1*(np.sqrt(2/784)) #He Initialization
    W_i2 = np.random.randn(128,128)
    W2 = W_i2*(np.sqrt(2/128)) #He Initialization
    W_i3 = np.random.randn(128,10)
    W3 = W_i3*(np.sqrt(2/128)) #He Initialization
    B1 = np.zeros((1,128))
    B2 = np.zeros((1,128))
    B3 = np.zeros((1,10))
    parameters = {
        "W1": W1, "B1": B1,
        "W2": W2, "B2": B2,
        "W3": W3, "B3": B3
    }
    return parameters

def front_propogation(X:np.ndarray,parameters,encoded:np.ndarray):
    W1, B1 = parameters["W1"], parameters["B1"]
    W2, B2 = parameters["W2"], parameters["B2"]
    W3, B3 = parameters["W3"], parameters["B3"]
    #layer 2(Hidden)
    Z2 = X @ W1 + B1
    A2 = relu(Z2)
    #Layer 3(Hidden)
    Z3 = A2 @ W2 + B2
    A3 = relu(Z3)
    #Output Layer
    Zfinal = A3 @ W3 + B3
    Zsoft = softmax(Zfinal)
    Loss = Cross_EntropyLoss(Zsoft,encoded)
    storage1 = {
        "X": X,
        "Z2": Z2, "A2": A2,
        "Z3": Z3, "A3": A3,
        "Zsoft": Zsoft,
        "Loss": Loss
    }
    return storage1

def back_propogation(encoded:np.ndarray,parameters,storage1):
    W1, B1 = parameters["W1"], parameters["B1"]
    W2, B2 = parameters["W2"], parameters["B2"]
    W3, B3 = parameters["W3"], parameters["B3"]
    X = storage1["X"]
    Z2, A2 = storage1["Z2"], storage1["A2"]
    Z3, A3 = storage1["Z3"], storage1["A3"]
    #Output Layer
    dZfinal = storage1["Zsoft"] - encoded
    dW3 = (A3.T @ dZfinal)/Batch_Size
    dB3 = np.sum(dZfinal,axis=0,keepdims=True)/Batch_Size
    #Layer 3(Hidden)
    dA3 = dZfinal @ W3.T
    dZ3 = dA3 * (Z3 > 0)
    dW2 = (A2.T @ dZ3)/Batch_Size
    dB2 = np.sum(dZ3,axis=0,keepdims=True)/Batch_Size
    #Layer 2(Hidden)
    dA2 = dZ3 @ W2.T
    dZ2 = dA2 * (Z2 > 0)
    dW1 = (X.T @ dZ2)/Batch_Size
    dB1 = np.sum(dZ2,axis=0,keepdims=True)/Batch_Size
    gradients = {
        "dW1": dW1, "dB1": dB1,
        "dW2": dW2, "dB2": dB2,
        "dW3": dW3, "dB3": dB3
    }
    return gradients

def update_parameters(parameters,gradients,learning_rate = 0.08):
    parameters["W1"] -= learning_rate * gradients["dW1"]
    parameters["B1"] -= learning_rate * gradients["dB1"]
    parameters["W2"] -= learning_rate * gradients["dW2"]
    parameters["B2"] -= learning_rate * gradients["dB2"]
    parameters["W3"] -= learning_rate * gradients["dW3"]
    parameters["B3"] -= learning_rate * gradients["dB3"]
    return parameters

def Test_Model(X_test:np.ndarray,parameters):
    W1, B1 = parameters["W1"], parameters["B1"]
    W2, B2 = parameters["W2"], parameters["B2"]
    W3, B3 = parameters["W3"], parameters["B3"]
    #layer 2(Hidden)
    Z2 = X_test @ W1 + B1
    A2 = relu(Z2)
    #Layer 3(Hidden)
    Z3 = A2 @ W2 + B2
    A3 = relu(Z3)
    #Output Layer
    Zfinal = A3 @ W3 + B3
    Zsoft = softmax(Zfinal)
    return Zsoft

    
    




