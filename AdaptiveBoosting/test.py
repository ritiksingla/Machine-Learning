from sklearn import datasets
from sklearn.model_selection import train_test_split
from AdaBoostClassifier import AdaBoostClassifier
from AdaBoostRegressor import AdaBoostRegressor

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

clf = AdaBoostClassifier(n_estimators=50, algorithm='SAMME')
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
regr = AdaBoostRegressor(loss='square')
regr.fit(X_train, y_train)
print(
    'Training R2 Score: {:.2f}\nTest R2 Score: {:.2f}'.format(
        regr.score(X_train, y_train), regr.score(X_test, y_test)
    )
)
