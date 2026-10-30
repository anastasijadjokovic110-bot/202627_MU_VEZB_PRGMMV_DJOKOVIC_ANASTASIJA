import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin


class PolynomialRegression(BaseEstimator, RegressorMixin):

    def __init__(self, degree=2, reg_lambda=0):
        self.degree = degree
        self.reg_lambda = reg_lambda
        self.coef_ = None

    def fit(self, X, y):

        X = np.asarray(X).reshape(-1)
        y = np.asarray(y).reshape(-1)

        Phi = np.vander(X, N=self.degree + 1, increasing=True)

        A = Phi.T @ Phi
        b = Phi.T @ y

        I = np.eye(self.degree + 1)

        A = A + self.reg_lambda * I

        self.coef_ = np.linalg.solve(A, b)

        return self

    def predict(self, X):

        X = np.asarray(X).reshape(-1)

        Phi = np.vander(X, N=self.degree + 1, increasing=True)

        return Phi @ self.coef_

    def score(self, X, y):

        y = np.asarray(y).reshape(-1)

        y_pred = self.predict(X)

        mse = np.mean((y - y_pred) ** 2)

        return mse