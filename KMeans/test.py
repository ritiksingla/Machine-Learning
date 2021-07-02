from KMeans import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.cm as cm

df = pd.read_csv('data.csv', header=None)
X = df.values

N_CLUSTERS = 3
kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=15).fit(X)

# Changing default colors of plt.scatterplot input data
colors = cm.rainbow(np.linspace(0, 1, N_CLUSTERS))
cs = [colors[x] for x in kmeans.labels_.tolist()]

# Plotting Input Data
plt.scatter(X[:, 0], X[:, 1], c=cs)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    color='black',
    marker='x',
    s=100,
    linewidths=2,
)
plt.show()
