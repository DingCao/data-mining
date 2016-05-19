"""module ALPHA

chooses the best ALPHA

"""
# !/usr/bin/python2
from os import system

import numpy as np

import params
# get params
from params import M_PARAM_TRAIN
from params import M_PARAM_VALIDATE
from params import N_FEATURE
from params import ALPHA
from params import LAMBDAS
from params import ITERS
from params import SPAN

from lr import train_lr_gd
from lr import lr_cost


def main():
    """  """
    print 'loading data...'
    XandYtrain = np.genfromtxt(params.X_TRAIN_FILE,
                               delimiter=',',
                               dtype='float',
                               skip_header=True)
    X_train = XandYtrain[:, 1:(params.N_FEATURE + 1)]
    y_train = XandYtrain[:, (params.N_FEATURE + 1)].reshape(params.M_PARAM_TRAIN, 1)

    XandY_val = np.genfromtxt(params.X_TRAIN_FILE,
                              delimiter=',',
                              dtype='float',
                              skip_header=True)
    X_val = XandY_val[:, 1:(params.N_FEATURE + 1)]
    y_val = XandY_val[:, (params.N_FEATURE + 1)].reshape(params.M_PARAM_VALIDATE, 1)
    print "Done!"

    error_train = []
    error_val = []
    i = 0
    for a_lambda in LAMBDAS:
        print 'lambda: %e' % a_lambda
        [cost, theta] = train_lr_gd('logistic', X_train, y_train, ALPHA,
                                    a_lambda, ITERS, 5)

        [cost_trained, grad] = lr_cost('logistic', X_train, y_train, theta)
        error_train.append(cost_trained)

        [cost_val, grad] = lr_cost('logistic', X_val, y_val, theta)
        error_val.append(cost_val)

        print 'trained cost: %f, valid cost: %f' % (error_train[i],
                                                    error_val[i])
        i += 1

    errors = np.transpose([error_train, error_val])
    np.savetxt(params.LAMBDA_FILE, errors)


if __name__ == '__main__':
    main()
    system("pause")
