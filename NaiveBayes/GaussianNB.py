import numpy as np


class GaussianNB:
    def __init__(self, *, priors=None):
        pass

    def update_mean_variance(self, n_past, mu, var, X, sample_weight=None):
        if X.shape[0] == 0:
            return mu, var
        if sample_weight is not None:
            n_new = float(sample_weight.sum())
            new_mu = np.average(X, axis=0, weights=sample_weight)
            new_var = np.average((X - new_mu) ** 2, axis=0, weights=sample_weight)
        else:
            n_new = X.shape[0]
            new_mu = np.mean(X, axis=0)
            new_var = np.var(X, axis=0)
        if n_past == 0:
            return new_mu, new_var
        n_total = n_past + n_new
        total_mu = (n_new * new_mu + n_past * mu) / n_total
        old_ssd = n_past * var
        new_ssd = n_new * new_var
        total_ssd = old_ssd + new_ssd + (n_new * n_past / n_total) * (mu - new_mu) ** 2
        total_var = total_ssd / n_total
        return total_mu, total_var

    def fit(self, X, Y, sample_weight=None):

        self.classes_ = np.unique(Y)
        n_features = X.shape[1]
        n_classes = len(self.classes_)

        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))

        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.class_prior_ = np.zeros(n_classes, dtype=np.float64)
        for i in self.classes_:
            X_i = X[Y == i, :]
            if sample_weight is not None:
                sw_i = sample_weight[Y == i]
                N_i = sample_weight.sum()
            else:
                sw_i = None
                N_i = X_i.shape[0]
            new_theta, new_sigma = self.update_mean_variance(
                self.class_count_[i], self.theta_[i, :], self.sigma_[i, :], X_i, sw_i
            )
            self.theta_[i, :] = new_theta
            self.sigma_[i, :] = new_sigma
            self.class_count_[i] += N_i
        self.class_prior_ = self.class_count_ / self.class_count_.sum()
        return self

    """The different naive Bayes classifiers differ
    mainly by the assumptions they make for n_ij"""

    def _joint_log_likelihood(self, X):
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = -0.5 * np.sum(np.log(2.0 * np.pi * self.sigma_[i, :]))
            n_ij -= 0.5 * np.sum(
                ((X - self.theta_[i, :]) ** 2) / (self.sigma_[i, :]), 1
            )
            joint_log_likelihood.append(jointi + n_ij)
        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def score(self, X, Y, sample_weight=None):
        from sklearn.metrics import accuracy_score

        return accuracy_score(Y, self.predict(X), sample_weight=sample_weight)
