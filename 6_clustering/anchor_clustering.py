import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import jqmcvi


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

object_sizes = pd.read_csv("data/object_sizes.csv")
# plt.scatter(x=object_sizes["width"], y=object_sizes["height"])

X = np.array(object_sizes[["width", "height"]]).reshape(-1, 2)

# print(X)

print("K-means:")

kmeans_clustering_model = KMeans(n_clusters=5)

kmeans_clustering_model.fit(X)

kmeans_classes = kmeans_clustering_model.predict(X)

ax1.set_title("K-means")
ax1.scatter(x=X[:, 0], y=X[:, 1], c=kmeans_classes, cmap="prism")

kmeans_centroids = [(int(round(x)), int(round(y))) for x, y in kmeans_clustering_model.cluster_centers_]
print(kmeans_centroids)

kmeanks_db_score = sm.davies_bouldin_score(X, kmeans_classes)
print("Davies-Bouldin score: {0} (less is better).".format(kmeanks_db_score))

kmeans_d_score = jqmcvi.dunn_fast(X, kmeans_classes)
print("Dunn score: {0} (more is better).".format(kmeans_d_score))

print("GMM:")

gmm_clustering_model = GaussianMixture(n_components=5)

gmm_clustering_model.fit(X)

gmm_classes = gmm_clustering_model.predict(X)

ax2.set_title("GMM")
ax2.scatter(x=X[:, 0], y=X[:, 1], c=gmm_classes, cmap="prism")

gmm_db_score = sm.davies_bouldin_score(X, gmm_classes)
print("Davies-Bouldin score: {0} (less is better).".format(gmm_db_score))

gmm_d_score = jqmcvi.dunn_fast(X, gmm_classes)
print("Dunn score: {0} (more is better).".format(gmm_d_score))

plt.show()
