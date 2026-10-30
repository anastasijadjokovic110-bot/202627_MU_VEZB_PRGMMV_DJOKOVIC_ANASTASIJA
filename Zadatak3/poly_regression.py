import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin


class PolynomialRegression(BaseEstimator, RegressorMixin):

    def __init__(self, degree=2):
        self.degree = degree
        self.coef_ = None

    def fit(self, X, y):

        X = np.asarray(X).reshape(-1)
        y = np.asarray(y).reshape(-1)

        self.coef_ = np.polynomial.polynomial.polyfit(
            X,
            y,
            self.degree
        )

        return self

    def predict(self, X):

        X = np.asarray(X).reshape(-1)

        return np.polynomial.polynomial.polyval(
            X,
            self.coef_
        )

    def score(self, X, y):

        y = np.asarray(y).reshape(-1)

        y_pred = self.predict(X)

        mse = np.mean((y - y_pred) ** 2)

        return mse