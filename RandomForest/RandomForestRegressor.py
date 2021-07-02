import numpy as np
import sys

sys.path.append('..')

from DecisionTree import DecisionTreeRegressor
from warnings import warn
from sklearn.metrics import r2_score


class RandomForestRegressor:
    def __init__(
        self,
        n_estimators=100,
        criterion='mse',
        min_samples_split=2,
        max_depth=None,
        max_features=None,
        bootstrap=True,
        oob_score=False,
    ):
        assert n_estimators > 0, 'n_estimators must be positive'
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        if oob_score == True:
            assert bootstrap == True, 'Bootstrap should be true for out of bag score!'
        self.oob_score = oob_score
        self.trees = []
        self.oob_samples = []

    def getBootstrappedData(self, X, Y):
        n_samples = X.shape[0]
        idx = np.random.choice(n_samples, n_samples)
        self.oob_samples.append(np.setdiff1d(np.arange(n_samples), np.unique(idx)))
        return (X[idx], Y[idx])

    def fit(self, X, Y, sample_weight=None):
        assert X.shape[0] == Y.shape[0]
        self.n_outputs_ = 1
        self.n_features_ = X.shape[1]

        self.trees = []
        for _ in range(self.n_estimators):
            if self.bootstrap == True:
                (X_, Y_) = self.getBootstrappedData(X, Y)
            else:
                (X_, Y_) = (X, Y)

            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X_, Y_, sample_weight)
            self.trees.append(tree)
        if self.oob_score == True:
            self.set_oob_score(X, Y)
        return self

    def set_oob_score(self, X, Y):
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        n_predictions = np.zeros(n_samples)
        self.oob_score_ = 0
        for n in range(self.n_estimators):
            unsampled_indices = self.oob_samples[n]
            predictions[unsampled_indices] += self.trees[n].predict(
                X[unsampled_indices]
            )
            n_predictions[unsampled_indices] += 1
        if (n_predictions == 0).any():
            warn(
                "Some inputs do not have OOB scores. "
                "This probably means too few trees were used "
                "to compute any reliable oob estimates."
            )
            n_predictions[n_predictions == 0] = 1

        predictions /= n_predictions
        self.oob_prediction_ = predictions
        self.oob_score_ = r2_score(Y, predictions)

    def predict(self, X):
        pred = np.zeros(X.shape[0])
        for n in range(self.n_estimators):
            tree = self.trees[n]
            pred += tree.predict(X)
        return pred / self.n_estimators

    def score(self, X, Y, sample_weight=None):
        assert len(self.trees) != 0, 'Fit the model first'
        assert X.shape[0] == Y.shape[0]
        return r2_score(Y, self.predict(X), sample_weight=sample_weight)
