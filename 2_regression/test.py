import numpy
from mlinsights.mlmodel import QuantileLinearRegression
import matplotlib.pyplot as plt

N = 100
X = numpy.random.random(N)
eps1 = (numpy.random.random(N // 10 * 9) - 0.5) * 0.1
eps2 = (numpy.random.random(N // 10)) * 10
eps = numpy.hstack([eps1, eps2])
X = X.reshape(-1, 1)
Y = X.ravel() * 3.4 + 5.6 + eps

print(X.shape)
print(Y.shape)

clq = QuantileLinearRegression()
clq.fit(X, Y)
trend = clq.predict(X)

plt.scatter(X, Y)
plt.plot(X, trend, color="g")

plt.show()
