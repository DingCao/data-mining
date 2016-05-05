from numpy import *

# conputes the cost and grad for a lineaer regression
def LinearRegCost(X, y, theta, alambda):
    m = y.shape[0]       # number of training exmaples
    J = 0   # the cost
    grad = zeros(theta.shape)   # inital gradient

    # print(X[1, :])

    X = hstack([ones((m, 1)), X])    # NOTE: X matrix is without bias

    h = dot(X, theta)   # first gets the Hypothesis vector
    vecReg = vstack([0, theta[1:]])     # first gets the Hypothesis vector

    # print(X.shape, h.shape, y.shape, theta.shape)

    # computes cost and the gradient
    J = (1/(2*m)) * dot((h-y).T, (h-y))
    grad = (1/m) * dot(X.T, (h-y))

    # regularization
    J = J + (alambda/(2*m)) * dot(vecReg.T, vecReg)
    grad = grad + (alambda/m) * vecReg

    return [J, grad]


def GradientDescenting(X, y, theta, alambda, alpha, num_iters, span):
    print("training...")
    for i in range(1, num_iters+1):
        [J, grad] = LinearRegCost(X, y, theta, alambda)

        if i%span == 0 or i == num_iters:
            print("current iter: ", i, "/", num_iters,
                  "Cost: ", sqrt(J[0, 0]*2), end='\r')

        theta = theta - alpha*grad      # update the theta.

    print("\ntrained!")

    return [theta, J]
