""" main function for linear regression model

Copyright (c) huangjj27@SYSU (SNO: 13331087). ALL RIGHTS RESERVERD.

"""
#!/usr/bin/python2
from os import *
from numpy import *
from lr import train_lr_gd
import params


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
    [J, theta] = train_lr_gd('linear', Xtrain, ytrain, params.ALPHA,
                             params.LAMBDA, params.ITERS, params.SPAN)
    savetxt(params.COST_FILE, J)

    # predicting
    print "predicting..."
    savetxt(params.PRDICTION_FILE,
            hstack([arange(m).reshape((m, 1)), dot(
                hstack([ones((m, 1)), Ztest]), theta)]),
            delimiter=',',
            header='id,reference',
            comments='',
            fmt='%d,%f')
    print "prediction saved."


if __name__ == '__main__':
    main()
    system("pause")
