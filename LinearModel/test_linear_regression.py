from LinearRegression import LinearRegression
from SGDRegressor import SGDRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("house_price.csv")
X, y = df.drop('price', axis=1).values, df['price'].values

# Normalizing not required while using normal_equations (LinearRegression)
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=16
)


print("........................LinearRegression........................")
regr = LinearRegression()
regr.fit(X_train, y_train)
print(
    'Training R2 Score: {:.2f}\nTest R2 Score: {:.2f}'.format(
        regr.score(X_train, y_train), regr.score(X_test, y_test)
    )
)

print("........................SGDRegressor........................")
regr_sgd = SGDRegressor(eta0=1, penalty='l2', tol=10)
regr_sgd.fit(X_train, y_train)
print(
    'Training R2 Score: {:.2f}\nTest R2 Score: {:.2f}'.format(
        regr_sgd.score(X_train, y_train), regr_sgd.score(X_test, y_test)
    )
)


# Plot input data
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection='3d')
ax.scatter3D(X[:, 0], X[:, 1], y, marker='o')
ax.zaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x / 1000) + 'K')
)
ax.set_xlabel('Area per Square Feet')
ax.set_ylabel('Rooms')
ax.set_zlabel('Price')

# Plot decision boundary
xx = np.outer(np.linspace(-2, 2, 30), np.ones(30))
yy = xx.copy().T
zz = regr.intercept_ + regr.coef_[0] * xx + regr.coef_[1] * yy
ax.plot_surface(xx, yy, zz, alpha=0.8, color='red')
plt.title("Linear Regression Boundary")
plt.show()
