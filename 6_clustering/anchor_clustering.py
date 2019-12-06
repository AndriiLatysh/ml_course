import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics
import sklearn.cluster as sk_cluster
# import jqmcvi


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

object_sizes = pd.read_csv("data/object_sizes.csv")
# plt.scatter(x=object_sizes["width"], y=object_sizes["height"])

X = np.array(object_sizes[["width", "height"]]).reshape(-1, 2)

# print(X)

print("K-means:")

kmeans_clustering_model = sk_cluster.KMeans(n_clusters=5, init="random", n_init=1)

kmeans_clustering_model.fit(X)

kmeans_classes = kmeans_clustering_model.predict(X)

ax1.set_title("K-means")
ax1.scatter(x=X[:, 0], y=X[:, 1], c=kmeans_classes, cmap="prism")

kmeans_centroids = [(int(round(x)), int(round(y))) for x, y in kmeans_clustering_model.cluster_centers_]
print(kmeans_centroids)

kmeans_db_score = sk_metrics.davies_bouldin_score(X, kmeans_classes)
print("Davies-Bouldin score: {0} (less is better).".format(kmeans_db_score))

# kmeans_d_score = jqmcvi.dunn_fast(X, kmeans_classes)
# print("Dunn score: {0} (more is better).".format(kmeans_d_score))

kmeans_s_score = sk_metrics.silhouette_score(X, kmeans_classes)
print("Silhouette score: {} (more is better).".format(kmeans_s_score))

print("K-means++:")

kmeans_pp_clustering_model = sk_cluster.KMeans(n_clusters=5, init="k-means++")

kmeans_pp_clustering_model.fit(X)

kmeans_pp_classes = kmeans_pp_clustering_model.predict(X)

ax2.set_title("K-means++")
ax2.scatter(x=X[:, 0], y=X[:, 1], c=kmeans_pp_classes, cmap="prism")

kmeans_pp_db_score = sk_metrics.davies_bouldin_score(X, kmeans_pp_classes)
print("Davies-Bouldin score: {0} (less is better).".format(kmeans_pp_db_score))

# kmeans_pp_d_score = jqmcvi.dunn_fast(X, kmeans_pp_classes)
# print("Dunn score: {0} (more is better).".format(kmeans_pp_d_score))

kmeans_pp_s_score = sk_metrics.silhouette_score(X, kmeans_pp_classes)
print("Silhouette score: {} (more is better).".format(kmeans_pp_s_score))

plt.show()
