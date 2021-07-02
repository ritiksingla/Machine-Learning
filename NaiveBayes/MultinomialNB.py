import numpy as np


class MultinomialNB:
    def __init__(self, alpha=1.0):
        """The smoothing priors accounts for features not
        present in the learning samples and prevents zero
        probabilities in further computations"""
        self.alpha = alpha

    def _count(self, X, Y):
        self.feature_count_ += np.dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)

    def _update_feature_log_prob(self):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1)
        smoothed_cc = smoothed_cc.reshape(-1, 1)

        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)

    def _update_class_log_prior(self):
        n_classes = len(self.classes_)
        log_class_count = np.log(self.class_count_)
        self.class_log_prior_ = log_class_count - np.log(self.class_count_.sum())

    def _joint_log_likelihood(self, X):
        return np.dot(X, self.feature_log_prob_.T) + self.class_log_prior_

    def fit(self, X, Y, sample_weight=None):
        n_features = X.shape[1]
        from sklearn.preprocessing import LabelBinarizer

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(Y)
        self.classes_ = labelbin.classes_
        n_classes = len(self.classes_)
        if Y.shape[1] == 1:
            if n_classes == 2:
                Y = np.concatenate((1 - Y, Y), axis=1)
            else:  # degenerate case: just one class
                Y = np.ones_like(Y)
        if sample_weight is not None:
            Y = Y.astype(np.float64, copy=False)
            sample_weight = np.atleast_2d(sample_weight)
            Y *= sample_weight.T
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self._count(X, Y)
        self._update_feature_log_prob()
        self._update_class_log_prior()
        return self

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def score(self, X, Y, sample_weight=None):
        from sklearn.metrics import accuracy_score

        return accuracy_score(Y, self.predict(X), sample_weight=sample_weight)
