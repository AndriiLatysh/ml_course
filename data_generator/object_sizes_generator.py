import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

X, _ = make_blobs(n_samples=200, centers=8, n_features=2, shuffle=True, center_box=(30, 70), cluster_std=8)
df = pd.DataFrame(data=X, columns=["width", "height"])
plt.scatter(x=df["width"], y=df["height"])

df.to_csv("object_sizes.csv", index=False)

plt.show()
