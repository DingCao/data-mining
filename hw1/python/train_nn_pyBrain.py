""" main function for linear regression model

Copyright (c) huangjj27@SYSU (SNO: 13331087). ALL RIGHTS RESERVERD.

"""
#!/usr/bin/python2

import numpy as np

from os import *
from numpy import *
import params
import sys
from pybrain.structure import *
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet

def main():
    print "Loading data from %s..." % params.TRAIN_FILE,
    XandYtrain = genfromtxt(params.TRAIN_FILE,
                            delimiter=',',
                            dtype='float',
                            skip_header=True)
    Xtrain = XandYtrain[:, 1:(params.N_FEATURE + 1)]
    ytrain = XandYtrain[:, (params.N_FEATURE + 1)].reshape(
        params.M_TRAIN, 1)
    # ymin = np.min(ytrain)
    # ymax = np.max(ytrain)
    # ytrain = (ytrain-ymin)/(ymax-ymin)
    print "Done!"

    print "building neural network...",
    fnn = FeedForwardNetwork()

    inLayer = LinearLayer(params.N_FEATURE, name='inLayer')
    hiddenLayer = SigmoidLayer(params.N_HIDDEN, name='hiddenLayer')
    outLayer = LinearLayer(1, name='outLayer')

    fnn.addInputModule(inLayer)
    fnn.addModule(hiddenLayer)
    fnn.addOutputModule(outLayer)

    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)

    fnn.addConnection(in_to_hidden)
    fnn.addConnection(hidden_to_out)

    fnn.sortModules()
    print "Done!"

    # getting data
    print "preprocessing data(to trainning set and validation set)...",
    DS = SupervisedDataSet(params.N_FEATURE,1)

    for i in range(params.M_TRAIN):
        DS.addSample(Xtrain[i], ytrain[i])

    X = DS['input']
    Y = DS['target']
    print "Done!"

    # training
    print "trainning with neural nerwork..."
    trainer = BackpropTrainer(fnn, DS, verbose=True, learningrate=params.ALPHA)

    trainer.trainUntilConvergence(maxEpochs=params.ITERS, validationProportion=0.01)
    print "Done!"

    # savetxt(params.COST_FILE)

    # predicting
    print "loading test file...",
    XtestWithID = genfromtxt(params.TEST_FILE,
                             delimiter=',',
                             dtype='float',
                             skip_header=True)
    Xtest = XtestWithID[:, 1:(params.N_FEATURE + 1)]
    print "Done!"

    print "predicting..."
    J_test = np.zeros((params.M_TEST, 1))
    for i in range(params.M_TEST):
        J_test[i] = fnn.activate(Xtest[i])
        # J_test[i] = J*(ymax-ymin)+ymin
        if (i % params.SPAN_OUTER == 0) or (i is params.M_TEST - 1):
            print ("predicting sample: %d/%d\r" % (i, params.M_TEST - 1)),
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
