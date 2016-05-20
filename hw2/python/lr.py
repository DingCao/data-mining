""" logistic regression model

Copyright (c) huangjj27@SYSU (SNO: 13331087). ALL RIGHTS RESERVERD.

"""
import sys

from numpy import dot
from numpy import exp
from numpy import hstack
from numpy import log
from numpy import ones
from numpy import vstack
from numpy import zeros

from random import randint

from params import CONVERGED
from params import CONVERGED_COUNT


def sigmoid(z):
    return 1 / (1 + exp(-z))


def hypho(lrtype, X, theta, m):
    """computes hyphothesis for logistic regression or linear regression

    Args:
        lrtype: decides to use which regression. dafault is the linear one
        X:  the samples matrix to predict. a sample each row
        theta: the weights for X
        m: the number of samples

    Returns:
        hyphothesis: the prediction vector for X
    """
    hyphothesis = dot(X, theta)

    if lrtype == 'logistic':
        hyphothesis = sigmoid(hyphothesis)

    if m == 1:
        hyphothesis = hyphothesis[0, 0]

    return hyphothesis


def cost_linear(h, label, m):
    a_cost = (1.0 / (2 * m)) * dot((h - label).T, (h - label))
    return a_cost[0, 0]


def cost_logistic(h, label, m):
    """
    computes loss for logistic regression betwen given prediction vector h
    and lalbel vector with their size m. make sure their sizes are the same!

    Returns:
        a_cost:
    """
    a_cost = (-1.0 / m) * (dot(label.T, log(h)) + dot(
        (1 - label).T, log(1 - h)))
    return a_cost[0, 0]


cost = {"logistic": cost_logistic, "linear": cost_linear}


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

    vecReg = vstack([0, theta[1:]])  # then gets the Regularation vector
    X = hstack([ones((m, 1)), X])  # NOTE: X matrix is without bias

    # computes
    h = hypho(lrtype, X, theta, m)  # the hyphothesis vector
    J = cost[lrtype](h, label, m)  # the loss between labels and hyphothesis
    grad = (1.0 / m) * dot(X.T, (h - label))  # the grad for theta

    # regularization
    J = J + ((a_lambda + 0.0) / (2 * m)) * dot(vecReg.T, vecReg)
    grad = grad + (a_lambda + 0.0 / m) * vecReg

    return [J[0, 0], grad]


def train_lr_gd(lrtype,
                X_train,
                label,
                alpha,
                a_lambda=0,
                iters=200,
                span=1,
                batch=0):
    """train a lr model with gradient descenting

    Args:
        batch: subset size of the samples
        alpha: learning rate
        iters: times for training
        X_train: sample matrix
        lrtype: the type of lr regression.linear or logistic
        label: the samples' label vector
        a_lambda: regularation parameter
        span: each span write a data

    Returns:
        cost: the cost list of each iters
        grad: the gradient of theta

    """

    # get the shape of sample matrix
    m, n = X_train.shape

    theta = zeros((X_train.shape[1] + 1, 1))

    cost_list = []
    last_cost = 0
    break_count = 0
    for i in range(1, iters + 1):
        if 0 < batch <= m / 2:
            choosen = randint(0, m / batch - 1)  # choose a batch from X_train
            X_batch = X_train[choosen:choosen + batch]
            y_batch = label[choosen:choosen + batch]
            [J, grad] = lr_cost(lrtype, X_batch, y_batch, theta, a_lambda)
        else:
            [J, grad] = lr_cost(lrtype, X_train, label, theta, a_lambda)

        theta = theta - alpha * grad  # gradient descenting

        if abs(last_cost - J) < CONVERGED:
            break_count += 1
        else:
            break_count = 0

        if break_count >= CONVERGED_COUNT:
            break

        last_cost = J

        if i % span == 0 or i == iters:
            cost_list.append(J)
            sys.stdout.write('iter: %4d/%4d, cost: %f\r' % (i, iters, J))
    print ''

    return [cost_list, theta]
