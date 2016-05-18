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
    train_file = open(params.X_TRAIN_FILE, 'r')
    validation_file = open(params.X_VALIDATION_FILE, 'r')

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
    train_file.close()

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
    validation_file.close()

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
