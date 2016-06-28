#!/usr/bin/python2
"""module get_validation

generate data sets used for cross-validation

"""

#from os import *
from os import system

# get params
import params
from params import M_PARAM_TRAIN
from params import M_PARAM_VALIDATE


def get_validation(train_file_name, x_train_file_name, x_validation_file_name):
    """get dataset for cross-validation

    get a small samples set for checking the train effort, and see if there is a
    high bias or high variance in the trained model.

    Args:
        train_file_name: the name of original data set
        x_train_file_name: the name of the training set used for validation
        x_validation_file_name: the name of the cross-validation set

    """

    train_file = open(train_file_name, "r")
    x_train_file = open(x_train_file_name, "w+")
    x_validation_file = open(x_validation_file_name, "w+")

    train_file.readline()
    for i in range(M_PARAM_TRAIN+1):
        a_line = train_file.readline()
        x_train_file.write(a_line)

    for i in range(M_PARAM_VALIDATE):
        a_line = train_file.readline()
        x_validation_file.write(a_line)

    train_file.close()
    x_train_file.close()
    x_validation_file.close()


def main():
    """runs the function get_validation

    """

    get_validation(params.TRAIN_FILE, params.X_TRAIN_FILE,
                   params.X_VALIDATION_FILE)


if __name__ == '__main__':
    main()
    system("pause")
