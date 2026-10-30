import pandas as pd

from poly_regression import PolynomialRegression


podaci = pd.read_csv("podaci.csv")

X = podaci["X"].values
y = podaci["Y"].values


model = PolynomialRegression(degree=2)

model.fit(X, y)


print("Parametri modela:")
print(model.coef_)


predikcije = model.predict(X)

print("\nPredikcije:")
print(predikcije)


mse = model.score(X, y)

print("\nMSE:")
print(mse)