#!/usr/bin/python2
"""module get_validation

generate data sets used for cross-validation

"""
import params
import numpy as np

from os import system

# get params
from params import M_PARAM_TRAIN
from params import M_PARAM_VALIDATE
from params import N_FEATURE
from params import ALPHA
from params import LAMBDA
from params import ITERS
from params import SPAN_OUTER
from params import SPAN

from lr import train_lr_gd
from lr import lr_cost


def validate(X_train, y_train, X_val, y_val, alpha, a_lambda, iters, span):
    """ """
    error_train = []
    error_val = []

    for i in range(1, M_PARAM_TRAIN / span + 1):
        print 'process: %d/%d' % (i, M_PARAM_TRAIN / span)

        [cost, theta] = train_lr_gd('linear', X_train[0:(i * span), :],
                                    y_train[0:(i * span), :], alpha, a_lambda,
                                    iters, SPAN)
        [cost_trained, grad] = lr_cost('linear', X_train[0:(i * span), :],
                                       y_train[0:(i * span), :], theta)
        error_train.append(cost_trained)

        [cost_val, grad] = lr_cost('linear', X_val, y_val, theta)
        error_val.append(cost_val)

        print 'trained cost: %f, valid cost: %f' % (error_train[i - 1],
                                                    error_val[i - 1])

    return [error_train, error_val]


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

    X_val = XandYtrain[M_PARAM_TRAIN:(M_PARAM_TRAIN+M_PARAM_VALIDATE), 1:(N_FEATURE + 1)]
    y_val = XandYtrain[M_PARAM_TRAIN:(M_PARAM_TRAIN+M_PARAM_VALIDATE), (
        N_FEATURE + 1)].reshape(M_PARAM_VALIDATE, 1)

    print 'data loaded.'

    print 'validating...'
    errors = validate(X_train, y_train, X_val, y_val, ALPHA, LAMBDA, ITERS,
                      SPAN_OUTER)

    errors = np.transpose(errors)
    np.savetxt(VALIDATED_FILE, errors)
    print 'validation done!'

if __name__ == '__main__':
    main()
    system("pause")
