import numpy as np
def featureNormalize(X):
    mu = X.mean(axis = 0)
    sigma = X.std(axis = 0)
    X = (X - mu)/sigma
    # Note: It's important to return mu and sigma while making prediction
    return [X, mu, sigma]