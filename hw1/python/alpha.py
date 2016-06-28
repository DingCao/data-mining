"""module alpha

chooses the best alpha

"""
# !/usr/bin/python2
from os import system

import numpy as np

import params
# get params
from params import M_PARAM_TRAIN
from params import N_FEATURE
from params import ALPHAS
from params import LAMBDA
from params import ITERS

from lr import train_lr_gd


def main():
    """  """
    print 'loading data...'
    XandYtrain = np.genfromtxt(params.TRAIN_FILE,
                               delimiter=',',
                               dtype='float',
                               skip_header=True)
    X_train = XandYtrain[0:M_PARAM_TRAIN, 1:(N_FEATURE + 1)]
    y_train = XandYtrain[0:M_PARAM_TRAIN, (N_FEATURE + 1)].reshape(
        M_PARAM_TRAIN, 1)

    print 'data loaded.'

    costs = []
    for alpha in ALPHAS:
        print 'alpha: %e' % alpha
        [cost, theta] = train_lr_gd('linear', X_train, y_train, alpha, LAMBDA,
                                    ITERS, 20)
        costs.append(cost)

    costs = np.transpose(costs)
    np.savetxt(params.ALPHA_FILE, costs)


if __name__ == '__main__':
    main()
    system("pause")
