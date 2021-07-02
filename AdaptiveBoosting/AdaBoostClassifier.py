import sys

sys.path.append('..')

import numpy as np
from DecisionTree import DecisionTreeClassifier

"""
Currently only apply SAMME and not SAMME.R algorithm
"""


class AdaBoostClassifier:
    def __init__(self, n_estimators=50, learning_rate=1.0, algorithm='SAMME'):
        assert n_estimators > 0, "n_estimators must be positive"
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.algorithm = algorithm

    def fit(self, X, Y):
        assert X.shape[0] == Y.shape[0], "Shapes for data and targets must be same"
        self.classes_ = np.unique(Y)
        self.n_classes_ = self.classes_.shape[0]
        self.estimators_ = []

        # Weight of estimator i make in final prediction
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)

        # Fraction of samples estimator i made wrong prediction
        self.estimator_errors_ = np.zeros(self.n_estimators, dtype=np.float64)

        n_samples = X.shape[0]
        sample_weight = np.full(n_samples, 1 / n_samples)
        # Main loop for AdaBoost
        for iboost in range(self.n_estimators):

            tree = DecisionTreeClassifier(max_depth=1)
            tree.fit(X, Y, sample_weight=sample_weight)
            self.estimators_.append(tree)

            pred = self.estimators_[iboost].predict(X)
            incorrect_mask = pred != Y

            # 1. Computing errors
            err = np.average(incorrect_mask, weights=sample_weight)
            self.estimator_errors_[iboost] = err
            if err <= 0:
                break
            # 2. Computing weights
            alpha = self.learning_rate * (
                np.log((1.0 - err) / err) + np.log(self.n_classes_ - 1.0)
            )
            self.estimator_weights_[iboost] = alpha
            if not iboost == self.n_estimators - 1:
                sample_weight *= np.exp(alpha * incorrect_mask)
                sample_weight_sum = np.sum(sample_weight)
                if sample_weight_sum <= 0:
                    break
                sample_weight /= sample_weight_sum
        return self

    def predict(self, X):
        classes = self.classes_[:, np.newaxis]
        p = np.sum(
            (estimator.predict(X) == classes).T * w
            for estimator, w in zip(self.estimators_, self.estimator_weights_)
        )
        p /= self.estimator_weights_.sum()
        return self.classes_.take(np.argmax(p, axis=1), axis=0)

    def score(self, X, Y, sample_weight=None):
        assert X.shape[0] == Y.shape[0]
        from sklearn.metrics import accuracy_score

        return accuracy_score(Y, self.predict(X), sample_weight=sample_weight)
