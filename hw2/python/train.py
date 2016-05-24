""" main function for logistic regression model

Copyright (c) huangjj27@SYSU (SNO: 13331087). ALL RIGHTS RESERVERD.

"""
# !/usr/bin/python2

from os import system

import numpy as np
import time

import sys

import lr

import params
from params import M_TEST
from params import N_REDUCED
from params import BATCH

from params import ALPHA
from params import UPDATE_RATE
# from params import ITERS
from params import SPAN
# from params import BATCH
from params import LAMBDA

from lr import train_lr_gd
from lr import lr_cost


def main():
    print "training data..."
    train_file = open(params.REDUCED_FILE, 'r')

    flag = 1
    outer = 1
    theta = np.zeros((N_REDUCED + 1, 1))
    costs = []
    alpha = ALPHA
    cost = 0.0

    while flag:
        X_train = np.zeros((BATCH, N_REDUCED))
        y_train = np.zeros((BATCH, 1))

        for i in range(BATCH):
            a_line = train_file.readline().strip()
            if a_line != "":
                a_line = a_line.split(' ')  # seperate the data

                # get the label
                y_train[i] = int(a_line[0])
                a_line.pop(0)  # throw the label away

                for pair in a_line:
                    pair = pair.split(':')
                    X_train[i, int(pair[0]) - 1] = float(pair[1])
            else:
                flag = 0

        if flag:
            [cost, grad] = lr_cost('logistic', X_train, y_train, theta)
            theta = theta - alpha * grad
            # costs.append(cost)

            # if outer > 100: flag = 0
        if outer % SPAN == 0:
            alpha = alpha * UPDATE_RATE
            costs.append(cost)
            sys.stdout.write('processing: %6d, cost: %f, alpha: %.3e\r' %
                             (outer, cost, alpha))

        outer += 1
    print ''

    costs = np.transpose(costs)
    np.savetxt(params.COST_FILE, costs)
    theta_name = 'F:/Git_file/data-mining/hw2/theta/theta_alpha%.2f_update%.2f_lambda%.2f_batch%d_%s.txt' % (
        ALPHA, UPDATE_RATE, LAMBDA, BATCH,
        time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    np.savetxt(theta_name, theta)

    train_file.close()

    # predicting

    print "predicting..."
    test_file = open(params.TEST_FILE, 'r')
    prediction_name = 'F:/Git_file/data-mining/hw2/predictions/predict_alpha%.2f_update%.2f_lambda%.2f_batch%d_%s.txt' % (
        ALPHA, UPDATE_RATE, LAMBDA, BATCH, time.strftime("%Y-%m-%d_%H-%M-%S",
                                                         time.localtime()))
    predict_file = open(prediction_name, 'w')

    X = np.zeros((1, N_REDUCED + 1))
    predict_file.write('id,label\n')

    for i in range(M_TEST + 1):
        a_line = test_file.readline().strip()
        if a_line != "":
            a_line = a_line.split(' ')  # seperate the data

            # get the label
            id_test = int(a_line[0])
            a_line.pop(0)  # throw the label away

            for pair in a_line:
                pair = pair.split(':')
                X[0, int(pair[0])] = float(pair[1])

            h = lr.sigmoid(np.dot(X, theta))
            hypho = 0
            if h > 0.5:
                hypho = 1
            else:
                hypho = 0

            sys.stdout.write('predicting....%6d/%6d\r' % (i, M_TEST))
            predict_file.write('%d,%d\n' % (id_test, hypho))

    test_file.close()
    predict_file.close()

    print "\nprediction saved."


if __name__ == '__main__':
    main()
    system("pause")
