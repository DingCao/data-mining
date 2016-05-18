""" logistic regression model

Copyright (c) huangjj27@SYSU (SNO: 13331087). ALL RIGHTS RESERVERD.

"""
import sys

from numpy import dot
from numpy import zeros
from numpy import ones
from numpy import hstack
from numpy import vstack
from numpy import log
from numpy import exp


# from os import system

def sigmoid(z):
    return 1 / (1 + exp(-z))


def hyphothesis_logistic(X, theta, m):
    """ """
    hyphothesis = sigmoid(dot(X, theta))
    if m == 1: hyphothesis = hyphothesis[0, 0]

    return hyphothesis


def hyphothesis_linear(X, theta, m):
    """ """
    hyphothesis = dot(X, theta)
    if m == 1: hyphothesis = hyphothesis[0, 0]

    return hyphothesis


hypho = {
    "logistic": hyphothesis_logistic,
    "linear": hyphothesis_linear
}


def cost_linear(h, label, m):
    cost = (1.0 / (2 * m)) * dot((h - label).T, (h - label))
    return cost[0, 0]


def cost_logistic(h, label, m):
    cost = (-1.0 / m) * (dot(label.T, log(h)) + dot((1 - label).T, log(1 - h)))
    return cost[0, 0]


cost = {
    "logistic": cost_logistic,
    "linear": cost_linear
}


def lr_cost(lrtype, X, label, theta, a_lambda=0):
    """calculate the cost of a lr model

    Args:
        lrtype: the type of lr regression.linear or logistic
        X: the samples' feature matrix
        label: the samples' label vector
        theta: the weights of X
        a_lambda: regularation parameter

    Returns:
        J: the cost
        grad: the gradient of theta
    """

    m = label.shape[0]  # number of training exmaples

    # print label
    # system("pause")
    # print theta
    grad = zeros(theta.shape)  # inital gradient

    # print(X[1, :])

    X = hstack([ones((m, 1)), X])  # NOTE: X matrix is without bias

    h = hypho[lrtype](X, theta, m)  # first gets the Hypothesis vector
    vecReg = vstack([0, theta[1:]])  # then gets the Regularation vector

    # print(X.shape, h.shape, y.shape, theta.shape)

    # computes cost and the gradient
    J = cost[lrtype](h, label, m)
    grad = (1.0 / m) * dot(X.T, (h - label))

    # regularization
    J = J + ((a_lambda+0.0) / (2 * m)) * dot(vecReg.T, vecReg)
    grad = grad + (a_lambda+0.0 / m) * vecReg

    return [J[0, 0], grad]


def train_lr_gd(lrtype, X_train, label, alpha, a_lambda=0, iters=200, span=1):
    """train a lr model with gradient descenting

    Args:
        lrtype: the type of lr regression.linear or logistic
        X: the samples' feature matrix
        label: the samples' label vector
        theta: the weights of X
        a_lambda: regularation parameter
        span: each span write a data

    Returns:
        cost: the cost list of each iters
        grad: the gradient of theta

    """
    cost = []
    theta = zeros((X_train.shape[1]+1, 1))

    for iter in range(1, iters + 1):
        [J, grad] = lr_cost(lrtype, X_train, label, theta, a_lambda)
        theta = theta - alpha * grad  # gradient descenting

        if iter % span == 0 or iter == iters:
            cost.append(J)
            sys.stdout.write('iter: %4d/%4d, cost: %f\r' % (iter, iters, J))

    print ''

    return [cost, theta]

def train_lr_mbgd(lrtype, X_train, label, alpha, a_lambda, iters, batch):
    """ """
    cost = []
    theta = zeros((X_train.shape[1]+1, 1))

    for iter in range(1, iters + 1):
        [J, grad] = lr_cost(lrtype, X_train, label, theta, a_lambda)
        theta = theta - alpha * grad  # gradient descenting

        cost.append(J)

        sys.stdout.write('iter: %4d/%4d, cost: %f\r' % (iter, iters, J))

    print ''

    return [cost, theta]
