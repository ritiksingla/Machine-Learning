from sklearn import datasets
from sklearn.model_selection import train_test_split
from GaussianNB import GaussianNB
from MultinomialNB import MultinomialNB

iris = datasets.load_iris()
X = iris.data
Y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y
)

clf = MultinomialNB()
clf.fit(X_train, y_train)

print('Training Accuracy Score: {:.2f}'.format(clf.score(X_train, y_train)))
print('Test Accuracy Score: {:.2f}'.format(clf.score(X_test, y_test)))


clf2 = GaussianNB()
clf2.fit(X_train, y_train)
print('Training Accuracy Score: {:.2f}'.format(clf2.score(X_train, y_train)))
print('Test Accuracy Score: {:.2f}'.format(clf2.score(X_test, y_test)))
