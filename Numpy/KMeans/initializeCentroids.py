import numpy as np

def initializeCentroids(X, K):
	num_examples = X.shape[0]
	num_features = X.shape[1]
	idx = np.random.permutation(num_examples)
	return X[idx[0:K]]