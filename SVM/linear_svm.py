import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from featureNormalize import *
from gradientDescent import *

df = pd.read_csv('data.csv', header=None)
X, y = df.drop(columns=2).values, df[2].values.reshape(-1, 1)
X, mu, sigma = featureNormalize(X)
y = 2 * y - 1

# Learning Parameters
theta = np.zeros((X.shape[1], 1))
bias = 0
num_iters = 100
C = 1

theta, bias, J_history = gradientDescent(X, y, theta, bias, num_iters, alpha=0.001, C=C)

# Plot input data
plt.figure(figsize=(12, 12))
plt.scatter(X[:, 0], X[:, 1], c=y.tolist(), marker='o')

# Plot decision boundary
plot_x = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100).reshape(-1, 1)
plot_y_1 = (1 / theta[1]) * (1 - (theta[0] * plot_x) - bias)
plot_y_m1 = (1 / theta[1]) * (-1 - (theta[0] * plot_x) - bias)
plot_y_0 = (1 / theta[1]) * (0 - (theta[0] * plot_x) - bias)

plt.plot(plot_x, plot_y_0, color='red', label="Hyperplane WX + B = 0")
plt.plot(plot_x, plot_y_1, alpha=0.5, label="Hyperplane WX + B = 1")
plt.plot(plot_x, plot_y_m1, alpha=0.5, label="Hyperplane WX + B = -1")
plt.legend()
plt.title("Decision Boundary")
plt.show()

# Plot the hinge loss training loss curve
x = np.arange(num_iters)
plt.plot(x, J_history)
plt.xlabel("Epochs")
plt.ylabel("Hinge Loss")
plt.title("Hinge Loss vs Epochs")
plt.show()

y_pred = (np.dot(X, theta) + bias) > 0
y = (y + 1) / 2
print(f'Accuracy: {(1 - np.mean(np.abs(y_pred - y)))*100}%')
