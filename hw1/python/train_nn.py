""" main function for linear regression model

Copyright (c) huangjj27@SYSU (SNO: 13331087). ALL RIGHTS RESERVERD.

"""
#!/usr/bin/python2
import numpy as np

from os import *
from numpy import *
from lr import train_lr_gd
import params
import sys
from neural_network import nnCost
from neural_network import nnPredict


def main():
    print "Loading data..."

    XandYtrain = genfromtxt(params.TRAIN_FILE,
                            delimiter=',',
                            dtype='float',
                            skip_header=True)
    Xtrain = XandYtrain[:, 1:(params.N_FEATURE + 1)]
    ytrain = XandYtrain[:, (params.N_FEATURE + 1)].reshape(params.M_TRAIN, 1)

    XtestWithID = genfromtxt(params.TEST_FILE,
                             delimiter=',',
                             dtype='float',
                             skip_header=True)
    Xtest = XtestWithID[:, 1:(params.N_FEATURE + 1)]

    print "Done!"

    # training
    # [J, theta] = train_lr_gd('linear', Xtrain, ytrain, params.ALPHA,
    #                          params.LAMBDA, params.ITERS_TRAIN, params.SPAN)

    theta = np.random.rand(
        (params.N_FEATURE + 1) * params.N_HIDDEN + (params.N_HIDDEN + 1), 1)
    Js = []
    for i in range(params.M_TRAIN):
        J, grad = nnCost(Xtrain[i, :].T.reshape((params.N_FEATURE, 1)),
                         ytrain[i], theta, params.N_FEATURE, params.N_HIDDEN)
        theta = theta - params.ALPHA * grad

        if i % params.SPAN == 0:
            Js.append(np.sqrt(J * 2))
            sys.std.write("iter: %d, cost: %f\r" % (i, np.sqrt(J * 2)))

    savetxt(params.COST_FILE, Js)

    # predicting
    print "predicting..."
    J_test = np.zeros((params.M_TEST, 1))
    for i in range(params.M_TEST):
        J_test[i] = nnPredict(Xtest[i].T.reshape((params.N_FEATURE, 1)),
                                 theta, params.N_FEATURE, params.N_HIDDEN)
        if i % params.SPAN == 0:
            sys.stdout.write("predicting sample: %d\r" % i)

    savetxt(
        params.PREDICTION_FILE,
        hstack([arange(params.M_TEST).reshape((params.M_TEST, 1)), J_test]),
        delimiter=',',
        header='id,reference',
        comments='',
        fmt='%d,%f')
    print "prediction saved."


if __name__ == '__main__':
    main()
    system("pause")
