import numpy as np
from warnings import warn


class SGDRegressor:
    def __init__(
        self,
        loss='squared_loss',
        penalty='l2',
        alpha=1e-4,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        learning_rate='invscaling',
        eta0=0.01,
        power_t=0.25,
        n_iter_no_change=5,
        verbose=False,
    ):
        self.loss = loss
        self.penalty = penalty
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.eta0 = eta0
        self.power_t = power_t
        self.n_iter_no_change = n_iter_no_change
        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol

    def _compute_cost(self, X, Y):
        n_samples = X.shape[0]
        ordinary_least_square = np.power(X @ self.coef_ + self.intercept_ - Y, 2).sum()
        if self.penalty == 'l2':
            regularized_term = np.power(np.linalg.norm(self.coef_), 2)
        else:
            regularized_term = np.linalg.norm(self.coef_)
        return (1 / (2 * n_samples)) * ordinary_least_square + (
            self.alpha / (2 * n_samples)
        ) * regularized_term

    def _get_gradients(self, X, Y):
        n_samples = X.shape[0]
        slopes_ols = (1 / n_samples) * (X.T @ (X @ self.coef_ + self.intercept_ - Y))
        if self.penalty == 'l2':
            slopes_reg_term = (self.alpha / n_samples) * (self.coef_)
        elif self.penalty == 'l1':
            slopes_reg_term = (self.alpha / n_samples) * np.sign(self.coef_)
        else:
            raise ValueError('Loss not supported')
        dW = slopes_ols + slopes_reg_term
        if self.fit_intercept:
            db = (1 / n_samples) * np.sum(X @ self.coef_ + self.intercept_ - Y)
            return dW, db
        else:
            return dW, None

    def fit(self, X, Y):
        X = X.astype(np.float64, copy=False)
        Y = Y.astype(np.float64, copy=False)
        n_samples, n_features = X.shape

        self.coef_ = np.random.rand(n_features)
        self.intercept_ = np.zeros(1)

        t = 1.0
        self.n_iter_ = 0
        power_t = self.power_t
        eta0 = self.eta0
        no_improvement_count = 0
        best_loss = np.inf

        for epoch in range(self.max_iter):
            if self.verbose:
                print(f'-- Epoch {epoch + 1}')
            if self.shuffle:
                perm = np.random.permutation(n_samples)
                X = X[perm]
                Y = Y[perm]
            dW, db = self._get_gradients(X, Y)
            eta = eta0 / np.power(t, power_t)
            self.coef_ -= eta * dW
            if self.fit_intercept:
                self.intercept_ -= eta * db
            self.n_iter_ += 1
            t += 1
            loss = self._compute_cost(X, Y)
            if loss > best_loss - self.tol:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
                best_loss = loss
            if no_improvement_count >= self.n_iter_no_change:
                if self.verbose:
                    print(f'Convergence after {epoch + 1} epoch')
                break
            if self.verbose:
                print(f'Loss: {loss} at epoch {epoch + 1}')
        if self.n_iter_ == self.max_iter:
            print(
                "Maximum number of iteration reached before "
                "convergence. Consider increasing max_iter to "
                "improve the fit."
            )
        return self

    def predict(self, X):
        return np.ravel(X @ self.coef_ + self.intercept_)

    def score(self, X, y):
        from sklearn.metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred)
