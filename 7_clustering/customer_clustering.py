import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as sk_preprocessing
import sklearn.cluster as sk_clustering
import scipy.cluster.hierarchy as sp_clustering_hr


def set_printing_options():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)


set_printing_options()

users = pd.read_csv("data/customer_online_closing_store.csv")
users["return_rate"] = users["items_returned"] / users["items_purchased"]
users["average_price"] = users["total_spent"] / users["items_purchased"]

print(users[["average_price", "return_rate", "overall_rating"]])

X = np.array(users[["average_price", "return_rate", "overall_rating"]]).reshape(-1, 3)
min_max_scaler = sk_preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

print(X)

plt.title("Customer dendogram")

linkage_method = "ward"
dendogram = sp_clustering_hr.dendrogram(sp_clustering_hr.linkage(X, method=linkage_method))  # single complete average ward

agglomerative_model = sk_clustering.AgglomerativeClustering(n_clusters=4, linkage=linkage_method)

agglomerative_model.fit(X)

classes = agglomerative_model.labels_

users["class"] = classes

print(users[["average_price", "return_rate", "overall_rating", "class"]])

user_pivot_table = users.pivot_table(index="class",
                                     values=["average_price", "return_rate", "overall_rating", "customer_id"],
                                     aggfunc={"average_price": np.mean, "return_rate": np.mean,
                                              "overall_rating": np.mean, "customer_id": len})
print(user_pivot_table)

plt.show()