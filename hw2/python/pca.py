"""module pca

PCA stands for Principle Components Analysis

Copyright (c) huangjj27@SYSU(SNO: 13331087). ALL RIGHTS RESERVED.

"""
import numpy as np
import params
from numpy import shape
from numpy import dot
from numpy import mean
from numpy import zeros
from numpy import std
from params import M_PARAM_TRAIN
from params import M_PARAM_VALIDATE
from params import N_FEATURE
from numpy.linalg import svd

from lr import train_lr_gd


def normalize(X):
    """
    before running pca, normalizing feature is important.

    """
    mu = mean(X, axis=0)
    X_norm = X - mu

    sigma = std(X, axis=0)
    for i in range(len(sigma)):
        if sigma[i] != 0:
            X_norm[:][i] = X_norm[:][i] / sigma[i]
    return X_norm, mu, sigma


def pca(X):
    """
    runs principal component analysis on the dataset X

    Returns:
        U: the eigen vectors for matrix X
        S: the eigen  valuse for matrix X
    """
    [m, n] = X.shape  # m for rows and n for columns
    Sigma = (1.0 / m) * dot(X.T, X)  # covarience matirx for X
    [U, S, V] = svd(Sigma)

    return U, S


def find_k(X):
    [U, S] = pca(X)
    k = 0
    diff = 0
    for i in range(S.shape[1]):
        diff += S[i, i]
        if diff >= 0.99:
            k = i
            break

    return U, S, k


def projection(X, U, k):
    U_reduce = U[:][1:k]
    Z = dot(X, U_reduce)

    return Z


def main():
    train_file = open(params.X_TRAIN_FILE, 'r')

    X_train = np.zeros((M_PARAM_TRAIN, N_FEATURE))
    y_train = np.zeros((M_PARAM_TRAIN, 1))
    for i in range(M_PARAM_TRAIN):
        a_line = train_file.readline().strip()
        a_line = a_line.split(' ')  # seperate the data
        # get the label
        y_train[i] = int(a_line[0])

    [X_norm, mu, sigma] = normalize(X_train)
    [U, S, k] = find_k(X_norm)

    print 'get the reduced dimension: %d' % k

    theta = zeros((k + 1, 1))
    Z = projection(X_norm, U, k)

    val = []
    [cost, theta] = train_lr_gd('logistic', Z, y_train, params.ALPHA)
    val.append(cost)
    [cost, theta] = train_lr_gd('logistic', X_train, y_train, params.ALPHA)
    val.append(cost)

    val = np.transpose(val)
    np.savetxt(params.PCA_FILE, val)


if __name__ == "__main__":
    main()
