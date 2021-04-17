import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as sk_cluster


object_sizes = pd.read_csv("data/object_sizes.csv")
X = object_sizes[["width", "height"]]

kmeans_pp_model = sk_cluster.KMeans(n_clusters=5, init="k-means++")
kmeans_pp_model.fit(X)

kmeans_pp_classes = kmeans_pp_model.predict(X)
plt.scatter(x=object_sizes["width"], y=object_sizes["height"], c=kmeans_pp_classes, cmap="gist_rainbow")

kmeans_pp_centroids = kmeans_pp_model.cluster_centers_
plt.scatter(x=kmeans_pp_centroids[:, 0], y=kmeans_pp_centroids[:, 1], marker="X", color="k", s=100)

plt.show()
