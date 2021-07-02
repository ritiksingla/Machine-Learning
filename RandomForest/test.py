from sklearn import datasets
from sklearn.model_selection import train_test_split
from RandomForestClassifier import RandomForestClassifier
from RandomForestRegressor import RandomForestRegressor

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
    'n_estimators': 100,
    'max_features': 3,
    'max_depth': 4,
    'min_samples_split': 5,
    'oob_score': True,
}
clf = RandomForestClassifier(**clf_params)
clf.fit(X_train, y_train)
print('Out of Bag score {:.2f}'.format(clf.oob_score_))
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

regr_params = {
    'max_depth': 4,
    'max_features': 3,
    'min_samples_split': 10,
    'oob_score': True,
}
regr = RandomForestRegressor(**regr_params)
regr.fit(X_train, y_train)
print('Out of Bag score {:.2f}'.format(regr.oob_score_))
print('Training R2 Score: {:.2f}'.format(regr.score(X_train, y_train)))
print('Test R2 Score: {:.2f}'.format(regr.score(X_test, y_test)))
