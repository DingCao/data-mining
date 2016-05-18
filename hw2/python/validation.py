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
from params import ALPHA_INIT
from params import LAMBDA

from lr import train_lr_gd
from lr import lr_cost


def validate(X_train, y_train, X_val, y_val, alpha, a_lambda, iters, span):
    """ """
    error_train = []
    error_val = []

    for i in range(1, M_PARAM_TRAIN / span + 1):
        print 'process: %d/%d' % (i, M_PARAM_TRAIN / span)

        [cost, theta] = train_lr_gd('logistic', X_train[0:(i * span), :],
                                    y_train[0:(i * span), :], alpha, a_lambda,
                                    iters)
        [cost_trained, grad] = lr_cost('logistic', X_train[0:(i * span), :], y_train[0:(i * span), :], theta)
        error_train.append(cost_trained)

        [cost_val, grad] = lr_cost('logistic', X_val, y_val, theta)
        error_val.append(cost_val)

        print 'trained cost: %f, valid cost: %f' % (error_train[i - 1],
                                                    error_val[i - 1])

    return [error_train, error_val]


def main():
    """  """
    train_file = open(params.X_TRAIN_FILE, 'r')
    validation_file = open(params.X_VALIDATION_FILE, 'r')
    validated_file = open(params.VALIDATED_FILE, "w")

    X_train = np.zeros((M_PARAM_TRAIN, N_FEATURE))
    y_train = np.zeros((M_PARAM_TRAIN, 1))
    for i in range(M_PARAM_TRAIN):
        a_line = train_file.readline().strip()
        a_line = a_line.split(' ')  # seperate the data

        # get the label
        y_train[i] = int(a_line[0])
        a_line.pop(0)  # throw the label away

        for pair in a_line:
            pair = pair.split(':')
            X_train[i, int(pair[0]) - 1] = float(pair[1])

    X_val = np.zeros((M_PARAM_VALIDATE, N_FEATURE))
    y_val = np.zeros((M_PARAM_VALIDATE, 1))
    for i in range(M_PARAM_VALIDATE):
        a_line = validation_file.readline().strip()
        a_line = a_line.split(' ')  # seperate the data

        # get the label
        y_val[i] = int(a_line[0])
        a_line.pop(0)  # throw the label away

        for pair in a_line:
            pair = pair.split(':')
            X_val[i, int(pair[0]) - 1] = float(pair[1])

    train_file.close()
    validation_file.close()

    errors = validate(X_train, y_train, X_val, y_val,
                                        ALPHA_INIT, LAMBDA, params.ITERS,
                                        params.SPAN)
    np.transpose(errors)

    for error in errors:
        for a_error in error:
            validated_file.write('%f ' % a_error)
        validated_file.write('\n')

    validated_file.close()


if __name__ == '__main__':
    main()
    system("pause")
