import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc


def set_printing_options():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)


set_printing_options()

users = pd.read_csv("data/customer_online_closing_store.csv")
users["return_rate"] = users["items_returned"] / users["items_purchased"]
users["average_price"] = users["total_spent"] / users["items_purchased"]

print(users[["average_price", "return_rate", "overall_rating"]])

X = np.array(users[["average_price", "return_rate", "overall_rating"]]).reshape(-1, 3)
min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)

print(X)

plt.title("Customer dendogram")
dend = shc.dendrogram(shc.linkage(X, method="ward"))  # single complete average ward

agglomerative_model = AgglomerativeClustering(n_clusters=4, linkage="ward")

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
