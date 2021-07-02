import numpy as np


class Node:
    def __init__(
        self, level_: int, n_samples_: int, sample_mask_: np.ndarray, criterion_='mse'
    ):
        self.level = level_
        self.criterion = criterion_
        self.n_samples = n_samples_
        self.sample_mask = sample_mask_

        self.splitting_gain = -1.0
        self.value = 0.0

        if self.criterion == 'mse':
            self.mse: float = 0
        elif self.criterion == 'mae':
            self.mae: float = 0

        self.isLeaf: bool = False

        # Only valid if isLeaf is False
        self.split_point: any = None
        self.split_feature: int = -1
        self.left: Node = None
        self.right: Node = None

    def __str__(self):
        res = f'Level: {self.level}\n'

        if self.criterion == 'mae':
            res += f'Mean Absolute Error: {self.mae}\n'
        elif self.criterion == 'mse':
            res += f'Mean Squared Error: {self.mse}\n'

        res += f'Number of Samples: {self.n_samples}\n'

        res += f'Value: {self.value}\n'

        if not self.isLeaf:
            res += f'Split Feature: {self.split_feature}\n'
            res += f'Split Point: {self.split_point}\n'
        else:
            res += 'Leaf Node: True\n'
        res += '..................................\n'
        return res


class DecisionTreeRegressor:
    def __init__(
        self, criterion='mse', max_depth=None, max_features=None, min_samples_split=2
    ):
        self.root = None
        if criterion not in ('mse', 'mae', 'friedman_mse'):
            raise ValueError("criterion must be 'mse','mae', or 'friedman_mse'")
        self.criterion = criterion
        self.max_depth = max_depth
        assert max_features == None or max_features > 0
        self.max_features_ = max_features
        assert min_samples_split >= 2
        self.min_samples_split = min_samples_split

    def __str__(self):
        res = ""
        if self.root == None:
            return res
        q = []
        q.append(self.root)
        while not len(q) == 0:
            cur_node = q.pop(0)
            res += str(cur_node)
            if not cur_node.isLeaf:
                q.append(cur_node.left)
                q.append(cur_node.right)
        return res

    def compute_mse(self, y_orig, sample_weight):
        assert y_orig.shape[0] == sample_weight.shape[0]

        weighted_n_node_samples = sample_weight.sum()
        sum_total = np.sum(y_orig * sample_weight)
        sq_sum_total = np.sum(y_orig * y_orig * sample_weight)
        mse = sq_sum_total / weighted_n_node_samples
        mse -= (sum_total / weighted_n_node_samples) ** 2.0
        return mse

    def compute_mae(self, y_orig, sample_weight):
        weighted_n_node_samples = sample_weight.sum()
        mae = 0.0
        mae += np.sum(np.fabs(y_orig - np.median(y_orig)) * sample_weight)
        return mae / weighted_n_node_samples

    def compute_splitting_gain(
        self, cur_criterion, left_Y, right_Y, sample_weight_left, sample_weight_right
    ):
        p = left_Y.shape[0] / (left_Y.shape[0] + right_Y.shape[0])
        if self.criterion == 'mse':
            return (
                cur_criterion
                - p * self.compute_mse(left_Y, sample_weight_left)
                - (1 - p) * self.compute_mse(right_Y, sample_weight_right)
            )
        elif self.criterion == 'mae':
            return (
                cur_criterion
                - p * self.compute_mae(left_Y, sample_weight_left)
                - (1 - p) * self.compute_mae(right_Y, sample_weight_right)
            )
        elif self.criterion == 'friedman_mse':
            mean_left = np.average(left_Y, weights=sample_weight_left)
            mean_right = np.average(right_Y, weights=sample_weight_right)
            diff = mean_right - mean_left
            n_left = left_Y.shape[0]
            n_right = right_Y.shape[0]
            return (n_left * n_right * (diff ** 2)) / (n_left + n_right)

    def get_nth_feature_means(self, X, N):
        means = []
        values = np.unique(X[X[:, N].argsort(), N])
        if values.shape[0] == 1:
            means.append(values[0])
            return means
        for i in range(values.shape[0] - 1):
            means.append(np.mean([values[i], values[i + 1]]))
        return means

    def build_tree(self, X, Y, sample_weight):
        n_samples = X.shape[0]
        sample_mask = np.full(n_samples, True)
        self.root = Node(0, n_samples, sample_mask, self.criterion)
        queue = []
        queue.append(self.root)
        while len(queue) > 0:
            cur_node = queue[0]
            queue.pop(0)
            if self.criterion == 'mse':
                cur_node.value = np.mean(Y[cur_node.sample_mask])
                cur_criterion = self.compute_mse(
                    Y[cur_node.sample_mask], sample_weight[cur_node.sample_mask]
                )
                cur_node.mse = cur_criterion

            elif self.criterion == 'mae':
                cur_node.value = np.median(Y[cur_node.sample_mask])
                cur_criterion = self.compute_mae(
                    Y[cur_node.sample_mask], sample_weight[cur_node.sample_mask]
                )
                cur_node.mae = cur_criterion

            elif self.criterion == 'friedman_mse':
                cur_node.value = np.mean(Y[cur_node.sample_mask])
                cur_criterion = self.compute_mse(
                    Y[cur_node.sample_mask], sample_weight[cur_node.sample_mask]
                )

            if (
                (cur_criterion == 0)
                or (self.max_depth != None and cur_node.level == self.max_depth)
                or (cur_node.n_samples < self.min_samples_split)
            ):
                cur_node.isLeaf = True
                continue

            valid_split = False
            while valid_split == False:

                if self.max_features_ == None or self.max_features_ >= X.shape[1]:
                    cur_features = range(X.shape[1])
                else:
                    cur_features = np.random.choice(
                        X.shape[1], self.max_features_, replace=False
                    )

                for N in cur_features:
                    possible_split_points = self.get_nth_feature_means(
                        X[cur_node.sample_mask], N
                    )
                    for split_point in possible_split_points:
                        left_mask = X.take(N, axis=1) <= split_point
                        right_mask = np.logical_not(left_mask)
                        left_mask = left_mask & cur_node.sample_mask
                        right_mask = right_mask & cur_node.sample_mask
                        left_sum = left_mask.sum()
                        right_sum = right_mask.sum()
                        if left_sum == 0 or right_sum == 0:
                            continue
                        assert left_sum + right_sum == cur_node.n_samples

                        valid_split = True
                        cur_sg = self.compute_splitting_gain(
                            cur_criterion,
                            Y[left_mask],
                            Y[right_mask],
                            sample_weight[left_mask],
                            sample_weight[right_mask],
                        )
                        if cur_node.splitting_gain < cur_sg:

                            cur_node.splitting_gain = cur_sg
                            cur_node.split_point = split_point
                            cur_node.split_feature = N

                            cur_node.left = Node(
                                cur_node.level + 1, left_sum, left_mask, self.criterion
                            )
                            cur_node.right = Node(
                                cur_node.level + 1,
                                right_sum,
                                right_mask,
                                self.criterion,
                            )
            if valid_split == True:
                if cur_node.left == None:
                    print(cur_node.splitting_gain)
                queue.append(cur_node.left)
                queue.append(cur_node.right)

    def fit(self, X, Y, sample_weight=None):
        assert X.shape[0] == Y.shape[0], "Shapes for samples and targets must be same"
        if Y.ndim == 1:
            self.n_outputs_ = 1
        else:
            self.n_outputs_ = Y.shape[1]
        assert self.n_outputs_ == 1, "Works only for single output"
        if self.max_features_ != None and X.shape[1] < self.max_features_:
            self.max_features_ = X.shape[1]

        self.n_classes_ = len(np.unique(Y).tolist())
        self.n_features_ = X.shape[1]
        if sample_weight is None:
            sample_weight = np.full(X.shape[0], 1 / X.shape[0])
        assert (
            X.shape[0] == sample_weight.shape[0]
        ), "Shapes for sample_weight and samples must be same"
        Y.reshape(sample_weight.shape)

        # Build tree recursively (without actual recursion)
        self.build_tree(X, Y, sample_weight)
        return self

    def predict_one(self, X):
        assert self.root != None, "Fit the model first"
        if X.ndim == 1:
            X.reshape(-1, 1)
        cur_node = self.root
        while cur_node.isLeaf != True:
            split_feature = cur_node.split_feature
            split_point = cur_node.split_point
            if X[split_feature] <= split_point:
                cur_node = cur_node.left
            else:
                cur_node = cur_node.right
        return cur_node.value

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        res = np.zeros(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            res[i] = self.predict_one(X[i])
        return res

    def score(self, X, Y, sample_weight=None):
        assert self.root != None, "Fit the model first"
        assert X.shape[0] == Y.shape[0]
        from sklearn.metrics import r2_score

        return r2_score(Y, self.predict(X), sample_weight=sample_weight)
