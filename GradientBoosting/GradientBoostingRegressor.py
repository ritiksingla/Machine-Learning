import numpy as np
import sys

sys.path.append('..')
from sklearn.dummy import DummyRegressor
from DecisionTree import DecisionTreeRegressor


class GradientBoostingRegressor:
    def __init__(
        self,
        loss='ls',
        learning_rate=0.1,
        n_estimators=100,
        criterion='friedman_mse',
        min_samples_split=2,
        max_depth=3,
        max_features=None,
        verbose=False,
        n_iter_no_change=None,
        tol=1e-4,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.max_depth = max_depth
        self.verbose = verbose
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol

    def _loss(self, y, raw_predictions, sample_weight=None):
        if sample_weight is None:
            return np.mean((y - raw_predictions) ** 2)
        else:
            return (1 / sample_weight.sum()) * np.sum(
                sample_weight * ((y - raw_predictions) ** 2)
            )

    def _raw_predict_init(self, X):
        return self.init_.predict(X).astype(np.float64)

    def _negative_gradient(self, y, raw_predictions):
        return y - raw_predictions

    def _fit_stage(self, i, X, y, raw_predictions, sample_weight, sample_mask):
        original_y = y

        """
        Need to pass a copy of raw_predictions to negative_gradient()
        because raw_predictions is partially updated at the end of the loop
        in update_terminal_regions(), and gradients need to be evaluated at
        iteration i - 1.
        """

        raw_predictions_copy = raw_predictions.copy()
        residual = self._negative_gradient(y, raw_predictions_copy)
        tree = DecisionTreeRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
        )
        tree.fit(X, residual, sample_weight=sample_weight)
        # print(raw_predictions.shape, tree.predict(X).shape)
        raw_predictions += self.learning_rate * tree.predict(X)
        self.estimators_[i] = tree
        return raw_predictions

    def _fit_stages(self, X, y, raw_predictions, sample_weight):
        n_samples = X.shape[0]
        sample_mask = np.ones(n_samples, dtype=bool)
        n_inbag = n_samples
        for i in range(self.n_estimators):
            # fit next stage of trees
            raw_predictions = self._fit_stage(
                i, X, y, raw_predictions, sample_weight, sample_mask
            )
            self.train_score_[i] = self._loss(y, raw_predictions, sample_weight)
        return i + 1

    def fit(self, X, y, sample_weight=None):
        n_samples, self.n_features_ = X.shape
        self.init_ = DummyRegressor(strategy='mean')
        self.estimators_ = np.empty(self.n_estimators, dtype=object)
        self.train_score_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.init_.fit(X, y, sample_weight=sample_weight)
        raw_predictions = self._raw_predict_init(X)
        n_stages = self._fit_stages(X, y, raw_predictions, sample_weight)
        # change shape of arrays after fit (early-stopping or additional ests)
        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]

        self.n_estimators_ = n_stages
        return self

    def predict(self, X):
        raw_predictions = self._raw_predict_init(X)
        for i in range(self.n_estimators):
            tree = self.estimators_[i]
            raw_predictions += tree.predict(X) * self.learning_rate
        return raw_predictions

    def score(self, X, Y):
        assert self.estimators_ is not None, "Fit the model first"
        assert X.shape[0] == Y.shape[0]
        from sklearn.metrics import r2_score

        return r2_score(Y, self.predict(X))
