import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from featureNormalize import *
from gradientDescent import *
from normalEquation import *

# sns.set_theme()

# Linear Regression with two features
df = pd.read_csv("house_price.csv")
X, y = df.drop('price',axis=1).values, df['price'].values.reshape(-1, 1)

# Number of features and examples respectively
num_features = X.shape[1]
m = X.shape[0]

# Normalize the training data
X, mu, sigma = featureNormalize(X)

# Add intercept term to X
X = np.concatenate([np.ones((X.shape[0], 1)), X], axis = 1)

# Initialize parameters
theta = np.random.rand(num_features + 1, 1)
alpha = 0.01
num_iters = 2000

# Train the linear regression model
theta, J_history = gradientDescent(X, y, theta, num_iters, alpha)
print(f'Theta values using gradientDescent: {theta}')

# Making Prediction
X_test = np.array([1, (3137 - mu[0])/sigma[0], (3 - mu[1])/sigma[1]])
print(f'Actual price for 3137 sq feet area and 3 rooms: 579900')
print(f'Predicted: {np.dot(X_test, theta)}')

# Plot input data
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection = '3d')
ax.scatter3D(X[:,1], X[:,2], y, marker='o')
ax.zaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: '{:,.0f}'.format(x/1000) + 'K'))
ax.set_xlabel('Area per Square Feet')
ax.set_ylabel('Rooms')
ax.set_zlabel('Price')

# Plot decision boundary
xx = np.outer(np.linspace(-2,2,30), np.ones(30))
yy = xx.copy().T
zz = theta[0] + theta[1]*xx + theta[2]*yy
ax.plot_surface(xx, yy, zz, alpha=0.8, color='red')
plt.title("Linear Regression Boundary")
plt.show()

# Plot the MSE loss curve
x = np.arange(len(J_history))
plt.plot(x, J_history)
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error Loss")
plt.title("MSE Loss vs Epochs")
plt.show()

# Print the final MSE for linreg model
print(f'Mean Squared Error: {computeCost(X, y, theta)}')

# # Solve using normal equation
X, y = df.drop('price',axis=1).values, df['price'].values.reshape(-1, 1)

# Add intercept term to X
X = np.concatenate([np.ones((X.shape[0], 1)), X], axis = 1)
theta = normalEquation(X, y)
print(f'Theta values using normal equation: {theta}')

# Making Prediction
X_test = np.array([1, 3137, 3])
print(f'Actual price for 3137 sq feet area and 3 rooms: 579900')
print(f'Predicted: {np.dot(X_test, theta)}')