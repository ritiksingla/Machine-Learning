import numpy as np

def computeMeans(X, idx, K):
	M = X.shape[0]
	N = X.shape[1]
	means = np.zeros(shape=(K, N))
	for i in range(K):
		means[i] = np.mean(X[idx == i], axis = 0)
	return means