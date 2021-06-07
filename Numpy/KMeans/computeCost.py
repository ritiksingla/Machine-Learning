import numpy as np

def computeCost(X, idx, centroids):
	cost = 0
	K = centroids.shape[0]
	for i in range(K):
		cost += np.sum(np.sqrt(np.sum(np.power(X[idx == i] - centroids[i], 2), axis = 1)))
	return cost