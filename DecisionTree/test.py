from sklearn import datasets
from sklearn.model_selection import train_test_split
from DecisionTreeClassifier import DecisionTreeClassifier
from DecisionTreeRegressor import DecisionTreeRegressor


print("........................Classifier........................")
make_clf_params = {
    'n_samples': 100,
    'n_features': 4,
    'n_redundant': 0,
    'random_state': 0,
}
X, Y = datasets.make_classification(**make_clf_params)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y
)

clf_params = {
    'max_depth': 3,
    'max_features': 3,
    'min_samples_split': 5,
    'criterion': 'entropy',
}
clf = DecisionTreeClassifier(**clf_params)
clf.fit(X_train, y_train)

print('Training Accuracy Score: {:.2f}'.format(clf.score(X_train, y_train)))
print('Test Accuracy Score: {:.2f}'.format(clf.score(X_test, y_test)))

print("........................Regressor........................")
make_regr_params = {
    'n_samples': 100,
    'n_features': 4,
    'n_informative': 2,
    'random_state': 0,
}
X, Y = datasets.make_regression(**make_regr_params)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

regr = DecisionTreeRegressor(max_depth=3, criterion='friedman_mse')
regr.fit(X_train, y_train)
print(
    'Training R2 Score: {:.2f}\nTest R2 Score: {:.2f}'.format(
        regr.score(X_train, y_train), regr.score(X_test, y_test)
    )
)
