import numpy as np

def computeCost(X, y, theta,bias, C):
	# Hinge loss
	return (1 / 2) * np.dot(theta.T, theta).item() + C * ((np.maximum(0, 1 - y * (np.dot(X, theta) + bias))).sum())