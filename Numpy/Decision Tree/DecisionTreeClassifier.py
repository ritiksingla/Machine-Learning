import numpy as np

class Node:
	def __init__(self, level_:int,X_:np.ndarray, Y_:np.ndarray):
		self.level = level_
		self.X = X_
		self.Y = Y_
		
		self.class_distribution:dict = {}
		self.gini:float = 0
		self.isLeaf:bool = False

		# Only valid if isLeaf is False
		self.information_gain:float = -1
		self.split_point:any = None
		self.split_feature:int = -1
		self.left:Node = None
		self.right:Node = None

	def __str__(self):
		res = f'Level: {self.level}\n'
		res += f'Gini Impurity: {self.gini}\n'
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
	def __init__(self):
		self.root = None

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
		gini = 1
		for key in class_distribution:
			prob = class_distribution[key]/float(node.shape[0])
			gini -= (prob**2)
		return gini
	def compute_information_gain(self, cur_gini, left_Y, right_Y):
		p = left_Y.shape[0] / (left_Y.shape[0] + right_Y.shape[0])
		return cur_gini - p * self.compute_gini_impurity(left_Y) - (1 - p)*self.compute_gini_impurity(right_Y)
	def get_nth_feature_means(self,X, N):
		means = []
		values = np.unique(X[X[:, N].argsort(), N])
		for i in range(values.shape[0] - 1):
			means.append(np.mean([values[i], values[i + 1]]))
		return means


	def build_tree(self, cur_node):
		X = cur_node.X
		Y = cur_node.Y
		assert X.shape[0] == Y.shape[0], "Shapes for data and targets must be same"

		cur_gini = self.compute_gini_impurity(Y)
		cur_node.gini = cur_gini
		cur_node.class_distribution = self.get_class_distribution(Y)
		if cur_gini == 0:
			cur_node.isLeaf = True
			return
		for N in range(X.shape[1]):
			possible_split_points = self.get_nth_feature_means(X, N)
			for split_point in possible_split_points:
				mask = (X[:,N] <= split_point)
				not_mask = np.logical_not(mask)
				left_X, left_Y = X[mask], Y[mask]
				right_X, right_Y = X[not_mask], Y[not_mask]
				cur_ig = self.compute_information_gain(cur_gini, left_Y, right_Y)
				if cur_node.information_gain < cur_ig:					
					
					cur_node.information_gain = cur_ig
					cur_node.split_point = split_point
					cur_node.split_feature = N

					cur_node.left = Node(cur_node.level + 1, left_X, left_Y)
					cur_node.right = Node(cur_node.level + 1, right_X, right_Y)
		self.build_tree(cur_node.left)
		self.build_tree(cur_node.right)
		return
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
		self.root = Node(0, X, Y)
		self.build_tree(self.root)

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