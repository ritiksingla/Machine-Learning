import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sigmoid import *
from gradientDescent import *
from featureNormalize import *

df = pd.read_csv('admission.csv')
X, y = df.drop('admitted', axis = 1).values, df['admitted'].values.reshape(-1, 1)

X, mu, sigma = featureNormalize(X)

X = np.concatenate([np.ones((X.shape[0], 1)), X], axis = 1)
theta = np.zeros((X.shape[1], 1))

# Train the linear regression model
num_iters = 2000
alpha = 0.01
theta, J_history = gradientDescent(X, y, theta, num_iters, alpha)

# Plot the input data
plt.scatter(X[:,1], X[:,2], c = y.tolist());
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")

# Plot decision boundary
plot_x = np.array([np.min(X[:,1]) - 2, np.max(X[:,1]) + 2])
plot_y = (-1/theta[2])*(theta[1] * plot_x + theta[0])
plt.plot(plot_x, plot_y, color = 'red')
plt.title("Decision Boundary")
plt.show()

# Plot the BCE training loss curve
x = np.arange(num_iters)
plt.plot(x, J_history)
plt.xlabel("Epochs")
plt.ylabel("BCE Loss")
plt.title("BCE Loss vs Epochs")
plt.show()

# Final loss and accuracy
print(f'Final MSE loss: {computeCost(X, y, theta)}')

y_pred = np.array([sigmoid(np.dot(X, theta)) > 0.5], dtype = np.int32).reshape(y.shape)
acc = (1 - np.mean(np.abs(y - y_pred)))*100
print(f'Final accuracy: {acc}%')

# Metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred, target_names = ['Not Admitted', 'Admitted']))

# High Precision: Not many negative class examples predicted as positive class
# High Recall: Predicted most positive class respectively

y_pred_prob = (sigmoid(np.dot(X, theta))).reshape(y.shape)
fpr, tpr, threshold = roc_curve(y, y_pred_prob)

plt.plot([0, 1], [0, 1], 'k--', label = 'Random Classifier')
plt.plot(fpr, tpr, label = 'Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.legend()
plt.show();

from sklearn.metrics import roc_auc_score
print(f'ROC Area Under Curve score: {roc_auc_score(y, y_pred_prob)}')