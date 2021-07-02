import numpy as np


class Node:
    def __init__(
        self, level_: int, n_samples_: int, sample_mask_: np.ndarray, criterion_='gini'
    ):
        self.level = level_
        self.criterion = criterion_
        self.n_samples = n_samples_
        self.sample_mask = sample_mask_

        self.sum_total: np.ndarray

        if self.criterion == 'gini':
            self.gini: float = 0
        else:
            self.entropy: float = 0

        self.isLeaf: bool = False

        # Only valid if isLeaf is False
        self.information_gain: float = -1
        self.split_point: any = None
        self.split_feature: int = -1
        self.left: Node = None
        self.right: Node = None

    def __str__(self):
        res = f'Level: {self.level}\n'

        if self.criterion == 'gini':
            res += f'Gini Impurity: {self.gini}\n'
        else:
            res += f'Entropy: {self.entropy}\n'

        res += f'Class Distribution: {str(self.sum_total)}\n'

        if not self.isLeaf:
            res += f'Information Gain: {self.information_gain}\n'
            res += f'Split Feature: {self.split_feature}\n'
            res += f'Split Point: {self.split_point}\n'
        else:
            res += 'Leaf Node: True\n'
        res += '..................................\n'
        return res


class DecisionTreeClassifier:
    def __init__(
        self, criterion='gini', max_depth=None, max_features=None, min_samples_split=2
    ):
        self.root = None
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

    def get_class_distribution(self, y_orig, sample_weight):
        assert y_orig.shape[0] == sample_weight.shape[0]
        sum_total = np.zeros(self.n_classes_)
        for i in range(y_orig.shape[0]):
            sum_total[y_orig[i]] += sample_weight[i]
        return sum_total

    def compute_gini_impurity(self, y_orig, sample_weight):
        weighted_n_node_samples = sample_weight.sum()
        sum_total = self.get_class_distribution(y_orig, sample_weight)
        sq_count = 0.0
        for c in range(self.n_classes_):
            count_k = sum_total[c]
            sq_count += count_k * count_k
        gini = 1.0 - sq_count / (weighted_n_node_samples * weighted_n_node_samples)
        return gini

    def compute_entropy(self, y_orig, sample_weight):
        sum_total = self.get_class_distribution(y_orig, sample_weight)
        weighted_n_node_samples = sample_weight.sum()
        entropy = 0.0
        for c in range(self.n_classes_):
            count_k = sum_total[c]
            if count_k > 0.0:
                count_k /= weighted_n_node_samples
                entropy -= count_k * np.log(count_k)
        return entropy

    def compute_information_gain(
        self, cur_criterion, left_Y, right_Y, sample_weight_left, sample_weight_right
    ):
        p = left_Y.shape[0] / (left_Y.shape[0] + right_Y.shape[0])
        if self.criterion == 'gini':
            return (
                cur_criterion
                - p * self.compute_gini_impurity(left_Y, sample_weight_left)
                - (1 - p) * self.compute_gini_impurity(right_Y, sample_weight_right)
            )
        else:
            return (
                cur_criterion
                - p * self.compute_entropy(left_Y, sample_weight_left)
                - (1 - p) * self.compute_entropy(right_Y, sample_weight_right)
            )

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
            if self.criterion == 'gini':
                cur_criterion = self.compute_gini_impurity(
                    Y[cur_node.sample_mask], sample_weight[cur_node.sample_mask]
                )
                cur_node.gini = cur_criterion
            else:
                cur_criterion = self.compute_entropy(
                    Y[cur_node.sample_mask], sample_weight[cur_node.sample_mask]
                )
                cur_node.entropy = cur_criterion

            cur_node.sum_total = self.get_class_distribution(
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
                        cur_ig = self.compute_information_gain(
                            cur_criterion,
                            Y[left_mask],
                            Y[right_mask],
                            sample_weight[left_mask],
                            sample_weight[right_mask],
                        )
                        if cur_node.information_gain < cur_ig:

                            cur_node.information_gain = cur_ig
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
                queue.append(cur_node.left)
                queue.append(cur_node.right)

    def fit(self, X, Y, sample_weight=None):
        assert X.shape[0] == Y.shape[0], "Shapes for samples and targets must be same"
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
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
        return np.argmax(cur_node.sum_total)

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        res = np.zeros(X.shape[0], dtype=np.int32)
        for i in range(X.shape[0]):
            res[i] = self.predict_one(X[i])
        return res

    def score(self, X, Y, sample_weight=None):
        assert self.root != None, "Fit the model first"
        assert X.shape[0] == Y.shape[0]
        from sklearn.metrics import accuracy_score

        return accuracy_score(Y, self.predict(X), sample_weight=sample_weight)
