"""module neural network

the cost function and gradient trainning with a sample

"""
#!/usr/bin/python2
import numpy as np
from lr import sigmoid

def ReLU(vec):
    return np.maximum(0, vec)

active = {
    "sigmoid": sigmoid,
    "ReLU": ReLU
}

def a_layer(input_vec, theta, active_type):
    """
    computes output vector given a un active input vector without bias,
    the params matrix theta, and active function

    """
    # row_in, col_in = input_vec.shape
    # row_theta, col_theta = theta.shape

    # if not (row_in is col_theta):

    input_vec = np.vstack([1, input_vec])  # add bias element
    input_vec = active[active_type](input_vec)  # choose a active function.

    output_vec = np.dot(theta, input_vec)

    return output_vec

# TODO: compute a neural network cost and grad with back propagation
def nnCost:
    pass