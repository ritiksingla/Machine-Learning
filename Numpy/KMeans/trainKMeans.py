import numpy as np
import matplotlib.pyplot as plt
from initializeCentroids import *
from findClosestCentroid import *
from computeMeans import *
from computeCost import *

# returns centroids, centroids_history for best choice of K on basis of elbow curve
def trainKMeans(X, threshold = 100, max_clusters = 10):
	final_cost = -1
	update = True
	final_clusters:int
	final_centroids_history:list
	final_idx:np.ndarray
	costs = []
	for CLUSTERS in range(1, max_clusters + 1):
		# Centroids List for plotting history
		centroids_history = []

		# Initializing Centroids
		centroids = initializeCentroids(X, CLUSTERS)

		centroids_history.append(centroids)
		prev_cost = -1
		while(True):
			idx = findClosestCentroid(X, centroids)
			centroids = computeMeans(X, idx, CLUSTERS)
			cur_cost = computeCost(X,idx,centroids)
			if prev_cost == -1 or prev_cost > cur_cost:
				prev_cost = cur_cost
				centroids_history.append(centroids)
			else:
				break

		cost = computeCost(X,idx,centroids)
		costs.append(cost)
		if not update:
			continue
		if final_cost == -1 or final_cost > cost:
			if final_cost != -1:
				if (final_cost - cost > threshold):
					final_cost = cost
					final_clusters = CLUSTERS
					final_idx = idx
					final_centroids_history = centroids_history
				else:
					update = False
			else:
				final_cost = cost
				final_clusters = CLUSTERS
				final_idx = idx
				final_centroids_history = centroids_history
	# Plot inertia vs K graph
	xx = np.arange(1, max_clusters + 1)
	plt.plot(xx, costs, "-o")
	plt.xlabel('Number of Clusters')
	plt.ylabel('Inertia')
	plt.title('Elbow Curve')
	plt.show()
	return [final_clusters, final_idx, final_centroids_history, final_cost]