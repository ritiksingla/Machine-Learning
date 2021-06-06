import numpy as np
def computeCost(X, y, theta):
    m = X.shape[0]
    assert X.shape[1] == theta.shape[0] and y.shape[0] == X.shape[0]
    return (1 / (2 * m)) * (((np.dot(X, theta) - y)**2).sum())