from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from RandomForestClassifier import RandomForestClassifier

iris = datasets.load_iris()
X = iris.data
Y = iris.target.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42, stratify = Y)
forest = RandomForestClassifier(criterion = 'entropy', max_features = 3, max_depth = 4)
forest.fit(X_train, y_train)
print('Training Accuracy: {:.2f}%'.format(forest.accuracy(X_train, y_train)))
print('Test Accuracy: {:.2f}%'.format(forest.accuracy(X_test, y_test)))