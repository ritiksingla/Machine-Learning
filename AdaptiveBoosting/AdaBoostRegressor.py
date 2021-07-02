import sys

sys.path.append('..')

import numpy as np
from DecisionTree import DecisionTreeRegressor

"""
Currently only apply SAMME and not SAMME.R algorithm
"""


class AdaBoostRegressor:
    def __init__(self, n_estimators=50, learning_rate=1.0, loss='linear'):
        assert n_estimators > 0, "n_estimators must be positive"
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        if loss not in ('linear', 'square', 'exponential'):
            raise ValueError("loss must be 'linear', 'square' or 'exponential'")
        self.loss = loss

    def fit(self, X, Y):
        assert X.shape[0] == Y.shape[0], "Shapes for data and targets must be same"
        self.estimators_ = []

        # Weight of estimator i make in final prediction
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)

        # Fraction of samples estimator i made wrong prediction
        self.estimator_errors_ = np.zeros(self.n_estimators, dtype=np.float64)

        n_samples = X.shape[0]
        sample_weight = np.full(n_samples, 1 / n_samples)
        # Main loop for AdaBoost
        for iboost in range(self.n_estimators):

            tree = DecisionTreeRegressor(max_depth=3)
            tree.fit(X, Y, sample_weight=sample_weight)
            self.estimators_.append(tree)

            Y_pred = self.estimators_[iboost].predict(X)

            error_vector = np.fabs(Y_pred - Y)
            sample_mask = sample_weight > 0
            masked_sample_weight = sample_weight[sample_mask]
            masked_error_vector = error_vector[sample_mask]
            error_max = masked_error_vector.max()
            if error_max != 0:
                masked_error_vector /= error_max
            if self.loss == 'square':
                masked_error_vector **= 2
            elif self.loss == 'exponential':
                masked_error_vector = 1.0 - np.exp(-masked_error_vector)
            estimator_error = (masked_sample_weight * masked_error_vector).sum()
            if estimator_error <= 0:
                print("Perfect Match")
                self.estimator_errors_[iboost] = 0
                self.estimator_weights_[iboost] = 1
                break
            elif estimator_error >= 0.5:
                print("Worse than random guess")
                if len(self.estimators_) > 1:
                    self.estimators_.pop(-1)
                break
            # Adaboost.R2
            beta = estimator_error / (1.0 - estimator_error)
            self.estimator_errors_[iboost] = estimator_error
            self.estimator_weights_[iboost] = self.learning_rate * np.log(1.0 / beta)
            if iboost < self.n_estimators - 1:
                sample_weight[sample_mask] *= np.power(
                    beta, (1.0 - masked_error_vector) * self.learning_rate
                )
                sample_weight_sum = sample_weight.sum()
                if sample_weight_sum <= 0:
                    break
                sample_weight /= sample_weight_sum
        return self

    def predict(self, X):
        num_samples = X.shape[0]

        predictions = np.array([est.predict(X) for est in self.estimators_]).T
        sorted_idx = np.argsort(predictions, axis=1)

        # First row is for first example and there are 'n_estimators' columns
        weighted_cdf = np.cumsum(self.estimator_weights_[sorted_idx], axis=1)

        # compare to 1/2 times of last column of runnning sum to get weighted median
        # for each sample. Expand the weighted_cdf by np.newaxis for compare compatibility
        median_or_above = weighted_cdf >= (0.5 * weighted_cdf[:, -1][:, np.newaxis])

        # Get first weight for each estimator that is True for 'median_or_above' mask
        median_idx = median_or_above.argmax(axis=1)

        median_estimators = sorted_idx[np.arange(num_samples), median_idx]
        return predictions[np.arange(num_samples), median_estimators]

    def score(self, X, Y, sample_weight=None):
        assert X.shape[0] == Y.shape[0]
        from sklearn.metrics import r2_score

        return r2_score(Y, self.predict(X), sample_weight=sample_weight)
