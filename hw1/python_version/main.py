from os import *
from numpy import *
from LinearReg import *

def main():
    print("Loading data...", end='')
    m = 25000
    n = 384
    XandYtrain = genfromtxt('data/train.csv',delimiter=',', dtype='float', skip_header=True)
    Xtrain = XandYtrain[:,1:(n+1)]
    ytrain = XandYtrain[:, (n+1)].reshape(m, 1)

    XtestWithID = genfromtxt('data/test.csv',delimiter=',', dtype='float', skip_header=True)
    Xtest = XtestWithID[:, 1:(n+1)]

    alambda = 0.0
    alpha = 0.03
    num_iters = 10000
    span = 5

    print("Done!")

    print("Pre-processing data...", end='')
    Ztrain = hstack([Xtrain, Xtrain*Xtrain])
    Ztest = hstack([Xtest, Xtest*Xtest])
    theta = loadtxt("data/theta_py_iter_150k.txt", dtype='float').reshape((2*n+1, 1))
    # theta = zeros((X.shape[1]+1, 1))
    print("Done!")

    # training
    [theta, J] = GradientDescenting(Ztrain, ytrain, theta, alambda, alpha, num_iters, span)
    savetxt("data/theta_py_iter_150k.txt", theta)
    print("theta saved.")

    # predicting
    print("predicting...", end='')
    savetxt("data/predict_py_150000.csv",
            hstack([arange(m).reshape((m, 1)),
                    dot(hstack([ones((m,1)), Ztest]), theta)]),
            delimiter=',', header='id,reference', comments='',
            fmt='%d,%f')
    print("prediction saved.")


if __name__ == '__main__':
    main()
    system("pause")
