""" main function for logistic regression model

Copyright (c) huangjj27@SYSU (SNO: 13331087). ALL RIGHTS RESERVERD.

"""
#!/usr/bin/python2
import numpy as np
import time

import lr

import params
from params import M_TEST
from params import M_TRAIN
from params import N_REDUCED

from params import ALPHA
from params import UPDATE_RATE
from params import ITERS
from params import SPAN
from params import BATCH
from params import LAMBDA

from lr import train_lr_gd
from lr import lr_cost


def main():
    print "Loading data...",
    train_file = open(params.REDUCED_FILE, 'r')

    flag = 1
    outer = 1
    theta = np.zeros((N_REDUCED + 1, 1))

    X_train = np.zeros((400*BATCH, N_REDUCED), dtype=np.int8)
    y_train = np.zeros((400*BATCH, 1), dtype=np.int8)
    X_val = np.zeros((100*BATCH, N_REDUCED+1), dtype=np.int8)
    y_val = np.zeros((100*BATCH, 1), dtype=np.int8)

    for i in range(400*BATCH):
        a_line = train_file.readline().strip()
        if a_line != "":
            a_line = a_line.split(' ')  # separate the data

            # get the label
            y_train[i] = int(a_line[0])
            a_line.pop(0)  # throw the label away

            for pair in a_line:
                pair = pair.split(':')
                X_train[i, int(pair[0]) - 1] = int(pair[1])

    for i in range(100*BATCH):
        a_line = train_file.readline().strip()
        if a_line != "":
            a_line = a_line.split(' ')  # separate the data

            # get the label
            y_val[i] = int(a_line[0])
            a_line.pop(0)  # throw the label away

            X_val[i, 0] = 1
            for pair in a_line:
                pair = pair.split(':')
                X_val[i, int(pair[0])] = int(pair[1])
    print 'Done!'

    print 'training data...'
    costs, theta = train_lr_gd('logistic', X_train, y_train, ALPHA,
        iters=ITERS, span=SPAN, batch=BATCH, update_rate=UPDATE_RATE)

    costs = np.transpose(costs)
    np.savetxt(params.COST_FILE, costs)
    theta_name = 'F:/Git_file/data-mining/hw2/theta/theta_alpha%.2f_update%.2f_lambda%.2f_batch%d_%s.txt' % (
        ALPHA, UPDATE_RATE, LAMBDA, BATCH,
        time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    # np.savetxt(theta_name, theta)

    train_file.close()
    print 'train Done!'

    print 'validating...',
    accurate = 0
    for i in range(100*BATCH):

        h = lr.sigmoid(np.dot(X_val[i].reshape((1, N_REDUCED+1)), theta))
        hypho = 0
        if h >= 0.5:
            hypho = 1

        if hypho == y_val[i]:
            accurate += 1

    print 'accuracy: %f' % ((accurate+0.0)/(100*BATCH))
    print 'Done!'

    print "predicting...\r",
    test_file = open(params.REDUCE_TEST, 'r')
    predict_file = open(params.PREDICTION_FILE, 'w')

    X = np.zeros((1, N_REDUCED + 1))
    predict_file.write('id,label\n')

    for i in range(M_TEST + 1):
        a_line = test_file.readline().strip()
        if a_line != "":
            a_line = a_line.split(' ')  # separate the data

            # get the label
            id_test = int(a_line[0])
            a_line.pop(0)  # throw the label away

            X[0, 0] = 1
            for pair in a_line:
                pair = pair.split(':')
                X[0, int(pair[0])] = int(pair[1])

            h = lr.sigmoid(np.dot(X, theta))
            hypho = 0
            if h >= 0.5:
                hypho = 1
            else:
                hypho = 0

            if i % 1000 == 0 or i == M_TEST-1:
                print 'predicting... %6d/%6d\r' % (i, M_TEST-1),
            predict_file.write('%d,%d\n' % (id_test, hypho))

    test_file.close()
    predict_file.close()
    print ''
    print 'prediction saved.'


if __name__ == '__main__':
    main()
