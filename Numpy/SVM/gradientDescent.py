import numpy as np
from computeCost import *

def gradientDescent(X, y, theta,bias,epochs = 1000, alpha = 0.01, C = 1):
	J_history = []
	for epoch in range(1, epochs + 1):
		conditions = (y * (np.dot(X, theta) + bias) >= 1)
		dtheta = (np.sum((1 - conditions) * (C * y * X), axis = 0)).reshape(-1, 1)
		db = ((1 - conditions) * -C * y).sum()
		theta = theta + alpha * dtheta + alpha*theta
		bias = bias - alpha * db
		cost = computeCost(X, y, theta, bias, C)
		J_history.append(cost)
	return [theta,bias,J_history]