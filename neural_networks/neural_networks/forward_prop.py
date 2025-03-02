import numpy as np

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    assert A.shape == Z.shape
    return A, cache


def initialize_parameters_deep(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)
    for l in range(1,L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    assert Z.shape == (W.shape[0], A.shape[1])
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    Z, cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    else:
        A, activation_cache = relu(Z)
    c = (cache, activation_cache)
    assert A.shape == (W.shape[0], A_prev.shape[1])
    return A, c


def L_model_forward(X, parameters):
    L = len(parameters) // 2
    A = X
    caches = []
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev,
                                             parameters["W" + str(l)],
                                             parameters["b" + str(l)],
                                             'relu')
        caches.append(cache)
    AL, cache = linear_activation_forward(A,
                              parameters["W" + str(L)],
                              parameters["b" + str(L)],
                              'sigmoid')
    caches.append(cache)
    assert(AL.shape == (1,X.shape[1]))
    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -(1/m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL)))
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    return cost




