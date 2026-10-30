import pandas as pd
import numpy as np

from poly_regression import PolynomialRegression


podaci = pd.read_csv("podaci.csv")

X = podaci["X"].values
y = podaci["Y"].values

stepeni = [1, 2, 3, 5, 9]

X_novo = np.array([600, 900, 1300, 1600])


for M in stepeni:

    model = PolynomialRegression(degree=M)

    model.fit(X, y)

    y_pred = model.predict(X)

    mse = model.score(X, y)

    E = 0.5 * mse

    nove_predikcije = model.predict(X_novo)

    print(f"\nM = {M}")

    print("Parametri:")
    print(model.coef_)

    print("E(w):")
    print(E)

    print("MSE:")
    print(mse)

    print("Predikcije za nove X vrednosti:")

    for x, y_novo in zip(X_novo, nove_predikcije):
        print(f"X = {x} -> Y = {y_novo}")