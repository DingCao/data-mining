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

    XandYtrain = genfromtxt(params.X_TRAIN_FILE,
                            delimiter=',',
                            dtype='float',
                            skip_header=True)
    Xtrain = XandYtrain[:, 1:(params.N_FEATURE + 1)]
    ytrain = XandYtrain[:, (params.N_FEATURE + 1)].reshape(
        params.M_PARAM_TRAIN, 1)
    ymin = np.min(ytrain)
    ymax = np.max(ytrain)
    ytrain = (ytrain-ymin)/(ymax-ymin)

    print "Done!"

    # training
    # [J, theta] = train_lr_gd('linear', Xtrain, ytrain, params.ALPHA,
    #                          params.LAMBDA, params.ITERS_TRAIN, params.SPAN)

    theta = np.random.randn((params.N_FEATURE + 1) * params.N_HIDDEN + (
        params.N_HIDDEN + 1), 1)
    MSE = []
    for i in range(1, params.ITERS+1):
        J, grad = nnCost(Xtrain, ytrain, theta, params.N_FEATURE,
                         params.N_HIDDEN)
        theta = theta - params.ALPHA * grad

        if i % params.SPAN == 0 or i == params.ITERS:
            MSE.append(np.sqrt(J * 2))
            sys.stdout.write("iter: %d, MSE: %.6e\r" % (i, MSE[i-1]))
            # print ("iter: %d, cost: %f\r" % (i, np.sqrt(J * 2)))
    print '\naverage MSE: %.6e' % MSE[-1]

    savetxt(params.COST_FILE, MSE)

    # checking
    print "loading validation file..."
    XandYval = genfromtxt(params.X_VALIDATION_FILE,
                            delimiter=',',
                            dtype='float',
                            skip_header=False)
    Xval = XandYval[:, 1:(params.N_FEATURE + 1)]
    yval = XandYval[:, (params.N_FEATURE + 1)].reshape(
        params.M_PARAM_VALIDATE, 1)
    yval = (yval-ymin)/(ymax-ymin)

    print "checking...",
    Jval, grad = nnCost(Xval, yval, theta, params.N_FEATURE,
                         params.N_HIDDEN)
    print 'MSE: %f' % mean(np.sqrt(Jval*2))

    # predicting
    print "loading test file..."

    XtestWithID = genfromtxt(params.TEST_FILE,
                             delimiter=',',
                             dtype='float',
                             skip_header=True)
    Xtest = XtestWithID[:, 1:(params.N_FEATURE + 1)]

    print "predicting..."

    J_test = np.zeros((params.M_TEST, 1))
    for i in range(params.M_TEST):
        J = nnPredict(Xtest[i].T.reshape((params.N_FEATURE, 1)), theta,
                              params.N_FEATURE, params.N_HIDDEN)
        J_test[i] = J*(ymax-ymin)+ymin
        if (i % params.SPAN_OUTER == 0) or (i is params.M_TEST - 1):
            sys.stdout.write("predicting sample: %d/%d\r" %
                             (i, params.M_TEST - 1))
            # print ("predicting sample: %d/%d\r" % (i, params.M_TEST - 1))
    print ''

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
