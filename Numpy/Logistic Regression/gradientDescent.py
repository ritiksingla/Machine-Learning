import numpy as np
from computeCost import *

def gradientDescent(X, y, theta, epochs = 1000, alpha = 0.001):
	J_history = []
	m = X.shape[0]
	for epoch in range(1, epochs + 1):
		h_theta = sigmoid(np.dot(X, theta))
		h_theta = np.clip(h_theta, 0.000001, 0.999999)
		slopes = (1/m) * np.dot(X.transpose(), (h_theta - y))
		theta = theta - alpha * slopes
		cost = computeCost(X, y, theta)
		J_history.append(cost)
	return [theta, J_history]