import pandas as pd

from poly_regression import PolynomialRegression


podaci = pd.read_csv("podaci.csv")

X = podaci["X"].values
y = podaci["Y"].values


model_bez = PolynomialRegression(degree=2, reg_lambda=0)

model_bez.fit(X, y)

mse_bez = model_bez.score(X, y)


print("BEZ REGULARIZACIJE")

print("Parametri:")
print(model_bez.coef_)

print("MSE:")
print(mse_bez)


model_sa = PolynomialRegression(degree=2, reg_lambda=1)

model_sa.fit(X, y)

mse_sa = model_sa.score(X, y)


print("\nSA REGULARIZACIJOM")

print("Parametri:")
print(model_sa.coef_)

print("MSE:")
print(mse_sa)