import numpy as np
from warnings import warn


class KMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300, verbose=False, tol=1e-4):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol

    def _init_centroids(self, X):
        num_examples = X.shape[0]
        # Initialize clusters as random sampled 'K' training samples
        random_rows = np.random.permutation(num_examples)
        return X[random_rows[0 : self.n_clusters]]

    def _get_closest_centroids(self, X, centers):
        num_examples = X.shape[0]
        dist = np.zeros(shape=(num_examples, self.n_clusters), dtype=np.float64)
        for c_i in range(self.n_clusters):
            dist[:, c_i] = np.linalg.norm(X - centers[c_i], axis=1)
        return np.argmin(dist, axis=1)

    def _recompute_centroids(self, X, labels):
        num_examples, num_features = X.shape
        new_centers = np.zeros(shape=(self.n_clusters, num_features), dtype=np.float64)
        for c_i in range(self.n_clusters):
            new_centers[c_i] = np.mean(X[labels == c_i], axis=0)
        return new_centers

    def _compute_inertia(self, X, labels, centers):
        inertia = 0
        for c_i in range(self.n_clusters):
            X_c_i = X[labels == c_i]
            inertia += np.sum(np.linalg.norm(X_c_i - centers[c_i], axis=1))
        return inertia

    def _kmeans_single(self, X):
        n_samples = X.shape[0]

        centroids = self._init_centroids(X)
        if self.verbose:
            print("Initialization complete")
        centroids_old = centroids.copy()

        labels = np.full(n_samples, -1, dtype=np.int32)
        labels_old = labels.copy()
        for i in range(self.max_iter):
            labels = self._get_closest_centroids(X, centroids)
            centroids = self._recompute_centroids(X, labels)
            if self.verbose:
                inertia = self._compute_inertia(X, labels, centroids)
                print(f"Iteration {i}, inertia {inertia}")
            if np.array_equal(labels, labels_old):
                if self.verbose:
                    print(f"Converged at iteration {i}: strict convergence.")
                break
            else:
                center_shift_tot = ((centroids - centroids_old) ** 2).sum()
                if center_shift_tot <= self.tol:
                    if self.verbose:
                        print(
                            f"Converged at iteration {i}: center shift "
                            f"{center_shift_tot} within tolerance {self.tol}."
                        )
                break
            labels_old[:] = labels
            centroids_old[:, :] = centroids

        inertia = self._compute_inertia(X, labels, centroids)
        return labels, inertia, centroids, i + 1

    def fit(self, X):
        # subtract of mean of x for more accurate distance computations
        assert X.shape[0] > self.n_clusters, "Number of Clusters exceeds num_examples"
        X_mean = X.mean(axis=0)
        X = X - X_mean
        best_inertia = None
        for i in range(self.n_init):
            labels, inertia, centers, n_iter_ = self._kmeans_single(X)
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
                best_n_iter = n_iter_
        best_centers += X_mean
        X += X_mean
        distinct_clusters = len(set(best_labels))
        if distinct_clusters < self.n_clusters:
            warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters)
            )
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def predict(self, X):
        return self._get_closest_centroids(X, self.cluster_centers_)

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    def transform(self, X):
        assert hasattr(self, 'labels_'), 'Fit the model first or call fit_transform'
        n_samples = X.shape[0]
        X_new = np.zeros((n_samples, self.n_clusters))
        for c_i in range(self.n_clusters):
            X_new[:, c_i] = np.linalg.norm(X - self.cluster_centers_[c_i], axis=1)
        return X_new

    def fit_transform(self, X):
        return self.fit(X).transform(X)
