import numpy as np

def findClosestCentroid(X, centroids):
	M = X.shape[0]
	K = centroids.shape[0]
	dist = np.zeros(shape=(M, K))
	for i in range(K):
		dist[:,i] = np.sqrt(np.sum(np.power(X - centroids[i], 2), axis = 1))
	return dist.argmin(axis = 1)