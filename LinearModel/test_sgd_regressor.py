from SGDRegressor import SGDRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("house_price.csv")
X, y = df.drop('price', axis=1).values, df['price'].values
X = StandardScaler().fit_transform(X)
regr = SGDRegressor(eta0=1, penalty='l1', tol=10, verbose=True)
regr.fit(X, y)
print(regr.score(X, y))
