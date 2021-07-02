import numpy as np


class LinearRegression:
    def __init__(self, fit_intercept=True, normalize=False):
        self.fit_intercept = fit_intercept
        self.normalize = normalize

    def _normal_equation(self, X, y):
        return (np.linalg.inv(X.T @ X)) @ (X.T @ y)

    def fit(self, X, Y):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        X_new = X.copy()
        if self.normalize == True:
            X_new = (X_new - X.mean(axis=0)) / np.linalg.norm(X, axis=0)

        # Add intercept term to X
        if self.fit_intercept:
            X_new = np.concatenate([np.ones((X_new.shape[0], 1)), X_new], axis=1)
        solver = self._normal_equation(X_new, Y)
        if self.fit_intercept:
            self.intercept_, self.coef_ = solver[0], solver[1:]
        else:
            self.coef_ = solver
            self.intercept_ = 0.0
        return self

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        from sklearn.metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred)
