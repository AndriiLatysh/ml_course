import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


X, _ = make_blobs(n_samples=200, centers=5, n_features=2, shuffle=True, center_box=(30, 70), cluster_std=4)
object_sizes = pd.DataFrame(data=X, columns=["width", "height"])
plt.scatter(x=object_sizes["width"], y=object_sizes["height"])

object_sizes = object_sizes.applymap(lambda v: int(round(v)))
print(object_sizes)

plt.show()

object_sizes.to_csv("object_sizes.csv", index=False)
