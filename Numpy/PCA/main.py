import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def computeCovarianceMatrix(X):
    Y = X
    n_samples = X.shape[0]
    covariance_matrix = (1 / (n_samples - 1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))
    return np.array(covariance_matrix, dtype=float)

def transform(X):
	covariance_matrix = computeCovarianceMatrix(X)
	eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
	idx = eigenvalues.argsort()[::-1]
	eigenvalues = eigenvalues[idx]
	eigenvectors = np.atleast_1d(eigenvectors[:, idx])
	X_transformed = X.dot(eigenvectors)
	return X_transformed

# Word2Vector pretrained embeddings
vocab,embeddings = [], []
with open('glove.txt', 'rt', encoding="utf8") as fi:
    full_content = fi.read().strip().split('\n')
for i in range(len(full_content)):
    i_word = full_content[i].split(' ')[0]
    i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
    vocab.append(i_word)
    embeddings.append(i_embeddings)
vocab_npa = np.array(vocab)
embs_npa = np.array(embeddings)
viz_words = 200
embed_pca = transform(embs_npa[0: viz_words])

fig = plt.figure(figsize = (16, 16))
for idx in range(viz_words):
    plt.scatter(embed_pca[idx, 0], embed_pca[idx, 1], color = "steelblue")
    plt.annotate(vocab_npa[idx], (embed_pca[idx, 0], embed_pca[idx, 1]), alpha = 0.7)
plt.title("Word Vectors")
plt.show();

# Iris Dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target.reshape(-1, 1)

sns.set_theme()

X = StandardScaler().fit_transform(X)

X_transformed = transform(X)
df = pd.DataFrame(np.concatenate([X_transformed[:, 0].reshape(-1,1), X_transformed[:, 1].reshape(-1,1), Y], axis = 1), columns = ['PC1', 'PC2', 'Specie'])
sns.scatterplot(x='PC1', y='PC2', data = df, hue = 'Specie')
plt.title("Iris Dataset")
plt.show()

# ANSUR Dataset
df = pd.read_csv("ANSUR.csv")
non_numeric = ['BMI_class', 'Height_class', 'Gender', 'Component', 'Branch']
df_numeric = df.drop(non_numeric,axis = 1)

pca_features = transform(df_numeric.values)
df['x'] = pca_features[:, 0]
df['y'] = pca_features[:, 1]

# Color according to BMI Class
sns.scatterplot(x='x',y='y',data=df,hue = 'BMI_class')
plt.axis('off')
plt.show()

# Color according to Height Class
sns.scatterplot(x='x',y='y',data=df,hue = 'Height_class')
plt.axis('off')
plt.show()