import numpy as np
import sys

sys.path.append('..')

from DecisionTree import DecisionTreeClassifier
from warnings import warn


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators=100,
        criterion='gini',
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
        self.n_classes_ = len(np.unique(Y).tolist())
        self.n_features_ = X.shape[1]

        self.trees = []
        for _ in range(self.n_estimators):
            if self.bootstrap == True:
                (X_, Y_) = self.getBootstrappedData(X, Y)
            else:
                (X_, Y_) = (X, Y)

            tree = DecisionTreeClassifier(
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
        n_classes_ = self.n_classes_

        predictions = np.zeros((n_samples, n_classes_))
        for n in range(self.n_estimators):
            unsampled_indices = self.oob_samples[n]
            pred = self.trees[n].predict(X[unsampled_indices])
            predictions[unsampled_indices, pred] += 1

        if (predictions.sum(axis=1) == 0).any():
            warn(
                "Some inputs do not have OOB scores. "
                "This probably means too few trees were used "
                "to compute any reliable oob estimates."
            )
        self.oob_score_ = np.mean(Y == np.argmax(predictions, axis=1), axis=0)

    def predict(self, X):
        pred_list = []
        pred = np.zeros(X.shape[0], dtype=np.int32)
        for n in range(self.n_estimators):
            tree = self.trees[n]
            cur_pred = tree.predict(X)
            for i in range(X.shape[0]):
                if n == 0:
                    pred_list.append(dict())
                if cur_pred[i] not in pred_list[i]:
                    pred_list[i][cur_pred[i]] = 1
                else:
                    pred_list[i][cur_pred[i]] += 1
        for i in range(X.shape[0]):
            pred[i] = max(pred_list[i], key=pred_list[i].get)
        return pred

    def score(self, X, Y, sample_weight=None):
        assert len(self.trees) != 0, 'Fit the model first'
        assert X.shape[0] == Y.shape[0]
        from sklearn.metrics import accuracy_score

        return accuracy_score(Y, self.predict(X), sample_weight=sample_weight)
