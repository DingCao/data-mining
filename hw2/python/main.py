""" main function for logistic regression model

Copyright (c) huangjj27@SYSU (SNO: 13331087). ALL RIGHTS RESERVERD.

"""
# !/usr/bin/python2

import numpy as np
import time

import sys

import lr

import params
from params import M_PARAM_TRAIN
from params import M_TEST
from params import N_FEATURE
from params import BATCH

from params import ALPHA
from params import ITERS
from params import SPAN
from params import BATCH
from params import LAMBDA
from os import system

from lr import train_lr_gd
from lr import lr_cost


def main():
    print "Loading data..."
    train_file = open(params.TRAIN_FILE, 'r')

    flag = 1

    outer = 1
    theta = np.zeros((N_FEATURE + 1, 1))
    while flag:
        X_train = np.zeros((M_PARAM_TRAIN, N_FEATURE))
        y_train = np.zeros((M_PARAM_TRAIN, 1))
        costs = []
        for i in range(M_PARAM_TRAIN):
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
            [cost, theta] = train_lr_gd('logistic', X_train, y_train, ALPHA,
                                        LAMBDA, ITERS, SPAN, BATCH)
            costs.append(cost)

        # if outer > 1: flag = 0

        print 'processing: %d' % outer
        outer += 1

    costs = np.transpose(costs)
    np.savetxt(params.COST_FILE, costs)
    theta_name = 'F:/Git_file/data-mining/hw2/theta/theta_alpha%.2f_lambda%.2f_batch%d_%s.txt' % (
        ALPHA, LAMBDA, BATCH,
        time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    np.savetxt(theta_name, theta)

    train_file.close()

    # predicting

    print "predicting..."
    test_file = open(params.TEST_FILE, 'r')
    prediction_name = 'F:/Git_file/data-mining/hw2/predictions/predict_alpha%.2f_lambda%.2f_batch%d_%s.txt' % (
        ALPHA, LAMBDA, BATCH, time.strftime("%Y-%m-%d_%H-%M-%S",
                                            time.localtime()))
    predict_file=open(prediction_name, 'w')

    X = np.zeros((1, N_FEATURE+1))
    predict_file.write('id,label\n')

    for i in range(M_TEST):
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

            sys.stdout.write('predicting....%4d/%4d\r' % (i, M_TEST))
            predict_file.write('%d,%d\n' % (id_test, h >= 0.5))

    test_file.close()
    predict_file.close()

    print "\nprediction saved."


if __name__ == '__main__':
    main()
    system("pause")
