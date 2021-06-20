import numpy as np

class Node:
	def __init__(self, level_:int,X_:np.ndarray, Y_:np.ndarray, criterion = 'gini'):
		self.level = level_
		self.X = X_
		self.Y = Y_
		self.criterion = criterion
		
		self.class_distribution:dict = {}
		if self.criterion == 'gini':
			self.gini:float = 0
		else:
			self.entropy:float = 0
		self.isLeaf:bool = False

		# Only valid if isLeaf is False
		self.information_gain:float = -1
		self.split_point:any = None
		self.split_feature:int = -1
		self.left:Node = None
		self.right:Node = None

	def __str__(self):
		res = f'Level: {self.level}\n'

		if self.criterion == 'gini':
			res += f'Gini Impurity: {self.gini}\n'
		else:
			res += f'Entropy: {self.entropy}\n'
		
		res += f'Class Distribution: {self.class_distribution}\n'
		
		if not self.isLeaf:
			res += f'Information Gain: {self.information_gain}\n'
			res += f'Split Feature: {self.split_feature}\n'
			res += f'Split Point: {self.split_point}\n'
		else:
			res += 'Leaf Node: True\n'
		res += '..................................\n'
		return res

class DecisionTreeClassifier:
	def __init__(self, criterion = 'gini', max_depth = None, max_features = None):
		self.root = None
		self.criterion = criterion
		self.max_depth = max_depth
		assert max_features == None or max_features > 0
		self.max_features = max_features

	def get_class_distribution(self, node):
		class_distribution = {}
		for target in node:
			if target.item() not in class_distribution:
				class_distribution[target.item()] = 1
			else:
				class_distribution[target.item()] += 1
		return class_distribution

	def compute_gini_impurity(self, node):
		class_distribution = self.get_class_distribution(node)
		cur_features = len(class_distribution)
		gini = 1
		for key in class_distribution:
			prob = class_distribution[key]/float(node.shape[0])
			gini -= (prob**2)
		return gini

	def compute_entropy(self, node):
		class_distribution = self.get_class_distribution(node)
		total = 0
		for x in class_distribution:
			total += class_distribution[x]
		entropy = 0
		for x in class_distribution:
			entropy += (class_distribution[x] / total) * np.log2(class_distribution[x] / total)
		if entropy < 0:
			return -entropy
		else:
			return entropy

	def compute_information_gain(self, cur_criterion, left_Y, right_Y):
		p = left_Y.shape[0] / (left_Y.shape[0] + right_Y.shape[0])
		if self.criterion == 'gini':
			return cur_criterion - p * self.compute_gini_impurity(left_Y) - (1 - p)*self.compute_gini_impurity(right_Y)
		else:
			return cur_criterion - p * self.compute_entropy(left_Y) - (1 - p)*self.compute_entropy(right_Y)
	
	def get_nth_feature_means(self, X, N):
		means = []
		values = np.unique(X[X[:, N].argsort(), N])
		if values.shape[0] == 1:
			means.append(values[0])
			return means
		for i in range(values.shape[0] - 1):
			means.append(np.mean([values[i], values[i + 1]]))
		return means

	def build_tree(self, cur_node):
		X = cur_node.X
		Y = cur_node.Y

		if self.criterion == 'gini':
			cur_criterion = self.compute_gini_impurity(Y)
			cur_node.gini = cur_criterion
		else:
			cur_criterion = self.compute_entropy(Y)
			cur_node.entropy = cur_criterion

		cur_node.class_distribution = self.get_class_distribution(Y)
		if cur_criterion == 0 or (self.max_depth != None and cur_node.level == self.max_depth):
			cur_node.isLeaf = True
			return self

		valid_split = False
		while valid_split == False:
			
			if self.max_features == None or self.max_features >= X.shape[1]:
				cur_features = range(X.shape[1])
			else:
				cur_features = np.random.choice(X.shape[1], self.max_features, replace = False)
			
			for N in cur_features:
				possible_split_points = self.get_nth_feature_means(X, N)
				for split_point in possible_split_points:
					mask = (X[:, N] <= split_point)
					not_mask = np.logical_not(mask)
					left_sum = mask.sum()
					right_sum = not_mask.sum()
					if left_sum == 0 or right_sum == 0:
						continue
					valid_split = True					
					left_X, left_Y = X[mask], Y[mask]
					right_X, right_Y = X[not_mask], Y[not_mask]
					cur_ig = self.compute_information_gain(cur_criterion, left_Y, right_Y)
					if cur_node.information_gain < cur_ig:				
						
						cur_node.information_gain = cur_ig
						cur_node.split_point = split_point
						cur_node.split_feature = N
						
						cur_node.left = Node(cur_node.level + 1, left_X, left_Y, self.criterion)
						cur_node.right = Node(cur_node.level + 1, right_X, right_Y, self.criterion)
		
		self.build_tree(cur_node.left)
		self.build_tree(cur_node.right)
		return self

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

	def fit(self, X, Y):
		assert X.shape[0] == Y.shape[0], "Shapes for data and targets must be same"
		if self.max_features != None and X.shape[1] < self.max_features:
			self.max_features = X.shape[1]
		self.root = Node(0, X, Y, self.criterion)
		return self.build_tree(self.root)

	def predict_one(self, X):
		assert self.root != None, "Fit the model first"
		cur_node = self.root
		while cur_node.isLeaf != True:
			split_feature = cur_node.split_feature
			split_point = cur_node.split_point
			if X[split_feature] <= split_point:
				cur_node = cur_node.left
			else:
				cur_node = cur_node.right
		return cur_node.Y[0].item()

	def predict(self, X):
		if len(X.shape) == 1:
			X = np.expand_dims(X, axis = 0)
		res = []
		for i in range(X.shape[0]):
			res.append(self.predict_one(X[i]))
		return res

	def accuracy(self, X, Y):
		assert self.root != None, "Fit the model first"
		pred = self.predict(X)
		pred = np.array(pred).reshape(-1, 1)
		correct = (pred == Y).sum()
		return (correct / X.shape[0])*100

class RandomForestClassifier:
    def __init__(self, n_estimators = 100, criterion = 'gini', max_depth = None, max_features = None, bootstrap = True, oob_score = False):
    	assert n_estimators > 0, "n_estimators must be positive"
    	self.n_estimators = n_estimators
    	self.criterion = criterion
    	self.max_depth = max_depth
    	self.max_features = max_features
    	self.bootstrap = bootstrap
    	if oob_score == True:
    		assert bootstrap == True, "Bootstrap should be true for out of bag score!"
    	self.oob_score = oob_score
    	self.trees = []

    def getBootstrappedData(self, X, Y):
        idx = np.random.choice(self.n_outputs_, self.n_outputs_)
        self.indices = np.unique(idx)
        return X[idx], Y[idx]

    def fit(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        
        self.n_outputs_ = Y.shape[0]
        self.n_classes_ = len(np.unique(Y).tolist())
        self.n_features_ = X.shape[1]

        self.trees = []
        if self.bootstrap == True:
        	X_,Y_ = self.getBootstrappedData(X, Y)
        else:
        	self.indices = np.arange(self.n_outputs_)
        	X_,Y_ = X, Y
        for _ in range(self.n_estimators):
        	tree = DecisionTreeClassifier(criterion=self.criterion, max_depth = self.max_depth, max_features = self.max_features)
        	tree.fit(X_, Y_)
        	self.trees.append(tree)

    def predict(self, X):
    	pred_list = []
    	pred = []
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
    	for key in pred_list:
    		pred.append(max(key, key=key.get))
    	return pred

    def accuracy(self, X, Y):
    	assert len(self.trees) != 0, "Fit the model first"
    	assert X.shape[0] == Y.shape[0]
    	pred = self.predict(X)
    	pred = np.array(pred).reshape(-1, 1)
    	correct = (pred == Y).sum()
    	return (correct / X.shape[0])*100