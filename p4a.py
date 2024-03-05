import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['figure.figsize'] = (12.0, 9.0)

data = pd.read_csv("p4a.csv")

X = data.iloc[:, 0]
Y = data.iloc[:, 1]

plt.scatter (X, Y)
plt.show()

X_mean = np.mean(X)
Y_mean = np.mean(Y)
num, den = 0, 0
for i in range(len(X)):
    num += (X[i] - X_mean)*(Y[i] - Y_mean)
    den += (X[i] - X_mean)**2
m = num / den
c = Y_mean - m * X_mean
print(m, c)

Y_pred = m * X + c

plt.scatter(X, Y)
plt.scatter(X, Y_pred, color='red')
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color="green")
plt.show()
