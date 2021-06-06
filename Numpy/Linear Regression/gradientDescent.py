import numpy as np
from computeCost import *

def gradientDescent(X, y, theta, num_iters:int = 1500, alpha:float = 0.001):
    J_history = []
    m = X.shape[0]
    for num_iter in range(1, num_iters + 1):
        slopes = (1 / m) * (np.dot(X.transpose(), (np.dot(X, theta) - y)))
        theta = theta - alpha * slopes
        cost = computeCost(X, y, theta)
        J_history.append(cost)
    return [theta, J_history]