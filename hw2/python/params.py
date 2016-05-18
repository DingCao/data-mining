"""some constants will be used in the project

Copyright (c) huangjj27@SYSU (SNO: 13331087). ALL RIGHTS RESERVERD.

"""

# the number of samples used for chosing param and do the cross validation.
M_PARAM_TRAIN = 1000
M_PARAM_VALIDATE = 1000
N_FEATURE = 11392
BATCH = 100

# the number of samples used for trains and predictions.
M_TRAIN = 2177020
M_TEST = 220244
N_FEATURE = 11392

# intial params
ALPHA_INIT = 3e-2
ALPHAS = [3e-2, 1e-2, 3e-2, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
UPDATE_RATE = 0.99
LAMBDA = 0
LAMBDAS = [10, 3, 1, 0.3, 0.1, 0.02, 0.01, 0.003, 0.001, 0]

CURVE_POINTS = 1000
PRINT_EACH = 100

SPAN = 10
ITERS = 200

# some useful file name
TRAIN_FILE = "F:/Git_file/data-mining/hw2/data/train.txt"
X_TRAIN_FILE = "F:/Git_file/data-mining/hw2/data/x_train.txt"
X_VALIDATION_FILE = "F:/Git_file/data-mining/hw2/data/x_validation.txt"
VALIDATED_FILE = "F:/Git_file/data-mining/hw2/data/validated.txt"
ALPHA_FILE = "F:/Git_file/data-mining/hw2/data/alpha.txt"
COST_FILE = "F:/Git_file/data-mining/hw2/data/cost.txt"
