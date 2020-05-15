import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


rng = np.random.RandomState()
X = np.matmul(rng.rand(2, 2), rng.randn(2, 100)).T
# X[:, 0] = X[:, 0] + 2
# X[:, 1] = 10 * X[:, 1] + 7
plt.scatter(X[:, 0], X[:, 1])

plt.show()

df = pd.DataFrame(X, columns=["x", "y"])
df.to_csv("data/pca_demo_data.csv")
