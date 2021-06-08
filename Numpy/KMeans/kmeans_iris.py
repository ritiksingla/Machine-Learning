from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from trainKMeans import *

iris = datasets.load_iris()
X = iris.data

CLUSTERS, idx, centroids_history, cost = trainKMeans(X, max_clusters = 5, threshold = 20)

# Changing default colors of plt.scatterplot input data
colors = cm.rainbow(np.linspace(0, 1, CLUSTERS))
cs = [colors[x] for x in idx.tolist()]

# Plotting Input Data
plt.scatter(X[:,0], X[:,2], c = cs)
cur_centroids = centroids_history[-1]
plt.scatter(cur_centroids[:,0],cur_centroids[:,1],color='black',marker='x',s=100, linewidths=2)
plt.title(f"CLUSTERS = {CLUSTERS}")
plt.show()