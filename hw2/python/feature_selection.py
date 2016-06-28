""" module feature_selection

For the features given in the train.txt contain so many zero features and even
some features never appear in the trainning set, thus dorps thos features.

Copyright (c) huangjj27@SYSU (SNO: 13331087). ALL RIGHTS RESERVERD.

"""
import params
from params import N_FEATURE
from params import M_TRAIN

import numpy as np

def getMap():
    feature_map = [0] * N_FEATURE
    train_file = open(params.TRAIN_FILE, 'r')

    max_features = 0

    print 'choosing features...'
    out = 1
    while out:
        a_line = train_file.readline().strip()
        if a_line != "":
            a_line = a_line.split(' ')  # seperate the data

            # get the label
            #y_train = int(a_line[0])
            a_line.pop(0)  # throw the label away

            count = 0
            for pair in a_line:
                pair = pair.split(':')
                index = int(pair[0])
                feature_map[index - 1] += 1
                count += 1

            if count > max_features:
                max_features = count

            if out % 3000 == 0 or out == M_TRAIN:
                print 'sample: %d, sample_feature: %d, max_features: %d\r' % (out, count, max_features),
            out += 1
        else:
            out = 0
    print ''
    train_file.close()

    # gets reducing map file
    reduce_map = []
    count = 1
    for i in range(len(feature_map)):
        if feature_map[i] > 0 and feature_map[i] != params.M_TRAIN:
            reduce_map.append([i+1, count])
            count += 1

    print 'useful features: %d' % len(reduce_map)

    np.savetxt(params.FEATURE_MAP, reduce_map, fmt='%d %d')
    print 'map saved!'


def mapTrain():
    print 'mapping trainning data...'

    train_file = open(params.TRAIN_FILE, 'r')
    reduced_file = open(params.REDUCED_FILE, "w")
    reduce_map = np.loadtxt(params.FEATURE_MAP, dtype="int")
    out = 1
    while out:
        a_line = train_file.readline().strip()
        if a_line != "":
            a_line = a_line.split(' ')  # seperate the data

            # get the label
            reduced_file.write("%d" % int(a_line[0]))
            a_line.pop(0)  # throws the label away

            count = 0
            for pair in a_line:
                pair = pair.split(':')
                index = int(pair[0])

                while reduce_map[count][0] < (index):
                    count += 1

                if reduce_map[count][0] == (index):
                    reduced_file.write(' %d:%d' % (reduce_map[count][1], 1))

            reduced_file.write('\n')
            if out % 3000 == 0 or out == M_TRAIN:
                print 'mapping sample: %d\r' % out,
            out += 1
        else:
            out = 0
    print ''
    print 'mapped!'
    train_file.close()
    reduced_file.close()


def mapTest():
    print 'mapping test data...'
    test_file = open(params.TEST_FILE, "r")
    reduced_test_file = open(params.REDUCE_TEST, "w")
    reduce_map = np.loadtxt(params.FEATURE_MAP, dtype="int")

    out = 1
    while out:
        a_line = test_file.readline().strip()
        if a_line != "":
            a_line = a_line.split(' ')  # seperate the data

            # get the label
            reduced_test_file.write("%d" % int(a_line[0]))
            a_line.pop(0)  # throws the label away

            count = 0
            for pair in a_line:
                pair = pair.split(':')
                index = int(pair[0])

                while reduce_map[count][0] < (index):
                    count += 1

                if reduce_map[count][0] == (index):
                    reduced_test_file.write(' %d:%d' % (reduce_map[count][1], 1))

            reduced_test_file.write('\n')
            if out % 300 == 0 or out == M_TRAIN:
                print 'mapping sample: %d\r' % out,
            out += 1
        else:
            out = 0
    print ''
    print 'mapped!'
    test_file.close()
    reduced_test_file.close()


def main():
    #getMap()
    #mapTrain()
    mapTest()


if __name__ == '__main__':
    main()
