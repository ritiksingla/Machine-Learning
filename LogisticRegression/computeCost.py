import numpy as np
from sigmoid import *


def computeCost(X, y, theta):
    m = X.shape[0]
    h_theta = sigmoid(np.dot(X, theta))
    cost = (
        -1 / m * (np.dot(y.T, np.log(h_theta)) + np.dot((1 - y).T, np.log(1 - h_theta)))
    )
    return cost.item()
