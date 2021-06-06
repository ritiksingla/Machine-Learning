import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sigmoid import *
from gradientDescent import *
from featureNormalize import *

df = pd.read_csv('admission.csv')
X, y = df.drop('admitted', axis = 1).values, df['admitted'].values.reshape(-1, 1)

X, mu, sigma = featureNormalize(X)

X = np.concatenate([np.ones((X.shape[0], 1)), X], axis = 1)
theta = np.zeros((X.shape[1], 1))

# Train the linear regression model
num_iters = 2000
alpha = 0.01
theta, J_history = gradientDescent(X, y, theta, num_iters, alpha)

# Plot the input data
plt.scatter(X[:,1], X[:,2], c = y.tolist());
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")

# Plot decision boundary
plot_x = np.array([np.min(X[:,1]) - 2, np.max(X[:,1]) + 2])
plot_y = (-1/theta[2])*(theta[1] * plot_x + theta[0])
plt.plot(plot_x, plot_y, color = 'red')
plt.title("Decision Boundary")
plt.show()

# Plot the BCE training loss curve
x = np.arange(num_iters)
plt.plot(x, J_history)
plt.xlabel("Epochs")
plt.ylabel("BCE Loss")
plt.title("BCE Loss vs Epochs")
plt.show()

# Final loss and accuracy
print(f'Final MSE loss: {computeCost(X, y, theta)}')

y_pred = np.array([sigmoid(np.dot(X, theta)) > 0.5])
acc = (1 - np.mean(np.abs(y - y_pred)))*100
print(f'Final accuracy: {acc}%')