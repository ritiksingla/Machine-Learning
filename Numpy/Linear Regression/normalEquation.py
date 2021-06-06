import numpy as np
def normalEquation(X, y):
	return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))