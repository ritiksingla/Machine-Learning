import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from trainKMeans import *

# Loading Data
df = pd.read_csv('data.csv', header = None)
X = df.values

CLUSTERS,idx,centroids_history,cost = trainKMeans(X, max_clusters = 5)

# Changing default colors of plt.scatterplot input data
colors = cm.rainbow(np.linspace(0, 1, CLUSTERS))
cs = [colors[x] for x in idx.tolist()]

# Plotting Input Data
plt.scatter(X[:,0], X[:,1], c = cs)


# Plotting Centroids and their history
for i in range(len(centroids_history)):
	cur_centroids = centroids_history[i]
	plt.scatter(cur_centroids[:,0],cur_centroids[:,1],color='black',marker='x',s=100, linewidths=2)
	if i > 0:
		for j in range(CLUSTERS):
			plt.plot([cur_centroids[j][0], centroids_history[i - 1][j][0]],[cur_centroids[j][1], centroids_history[i - 1][j][1]],'k-', alpha=0.5)
plt.title(f"CLUSTERS = {CLUSTERS}")
plt.show()