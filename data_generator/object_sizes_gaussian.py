import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


number_of_clusters = 2
reshape_range = (1, 2)
X, y = make_blobs(n_samples=100, centers=number_of_clusters, n_features=2, shuffle=True, center_box=(30, 70), cluster_std=5)
multipliers = [(random.uniform(*reshape_range), random.uniform(*reshape_range)) for z in range(number_of_clusters)]
print(multipliers)

for z in range(len(X)):
    multiplier = multipliers[y[z]]
    X[z][0] *= multiplier[0]
    X[z][1] *= multiplier[1]

object_sizes = pd.DataFrame(data=X, columns=["width", "height"])
plt.scatter(x=object_sizes["width"], y=object_sizes["height"])

object_sizes = object_sizes.applymap(lambda v: int(round(v)))
print(object_sizes)

plt.show()

object_sizes.to_csv("object_sizes.csv", index=False)
