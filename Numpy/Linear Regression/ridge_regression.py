import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Ridge():
	def __init__(self, alpha = 1.0, normalize = False, max_iter = 1000, learning_rate = 0.01):
		self.alpha = alpha
		self.normalize = normalize
		self.max_iter = max_iter
		self.learning_rate = learning_rate
		self.history = []

	def normalize_data(self, X):
		mu = X.mean(axis = 0)
		sigma = X.std(axis = 0)
		X = (X - mu)/sigma
		return X

		# Does not work quite well!
		# X = ((X - np.mean(X, axis = 0)) /np.linalg.norm(X))
		# return X

	def computeCost(self, X, Y):
	    m = X.shape[0]
	    ordinary_least_square = (1 / (2 * m)) * (np.power(X@self.coef_ + self.intercept_ - Y, 2).sum())
	    regularized_term = (self.alpha / (2 * m)) * np.power(np.linalg.norm(self.coef_), 2)
	    return ordinary_least_square + regularized_term

	def gradientDescent(self, X, Y):
	    m = X.shape[0]
	    for num_iter in range(1, self.max_iter + 1):
	        
	        slopes_ols = (1 / m) * (X.T@(X@self.coef_ + self.intercept_ - Y))
	        slopes_reg_term = (self.alpha / m) * (self.coef_)
	        
	        dW = slopes_ols + slopes_reg_term
	        db = (1 / m) * np.sum(X@self.coef_ + self.intercept_ - Y)
	        
	        self.intercept_ = self.intercept_ - self.learning_rate * db
	        self.coef_ = self.coef_ - self.learning_rate * dW
	        
	        cost = self.computeCost(X, Y)
	        self.history.append(cost)

	def fit(self, X, Y):
		assert X.shape[0] == Y.shape[0], "Number of examples mismatched"
		m = X.shape[0]
		n = X.shape[1]
		self.coef_ = np.zeros((n, 1))
		self.intercept_ = 0.0
		if self.normalize == True:
			X = self.normalize_data(X)
		self.gradientDescent(X, Y)

	def predict(self, X):
		assert X.shape[1] == self.coef_.shape[0], "Number of features mismatched"
		if self.normalize == True:
			X = self.normalize_data(X)
		return X@self.coef_ + self.intercept_

	def score(self, X, Y):
		pred = self.predict(X)
		u = np.power(Y - pred, 2).sum()
		v = np.power(Y - Y.mean(), 2).sum()
		return (1 - u / v)

	def plot_history(self):
		x = np.arange(len(self.history))
		plt.plot(x, self.history)
		plt.xlabel("Epochs")
		plt.ylabel("Ridge Loss")
		plt.title("Ridge Loss vs Epochs")
		plt.show()

df = pd.read_csv("house_price.csv")
X, y = df.drop('price',axis=1).values, df['price'].values.reshape(-1, 1)
ridge = Ridge(learning_rate = 0.01, normalize = True, max_iter = 2000)
ridge.fit(X, y)
ridge.plot_history()
print(ridge.score(X, y))