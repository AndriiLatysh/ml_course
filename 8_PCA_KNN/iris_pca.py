import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as sk_preprocessing
import sklearn.decomposition as sk_decomposition
import sklearn.model_selection as sk_model_selection
import sklearn.linear_model as sk_linear

iris_df = pd.read_csv("data/iris.csv").sample(frac=1)
column_names = iris_df.columns.tolist()

X = iris_df[column_names[:-1]]
y = iris_df[column_names[-1]]

standard_scaler = sk_preprocessing.StandardScaler()
X = standard_scaler.fit_transform(X)

label_encoder = sk_preprocessing.LabelEncoder()
y = label_encoder.fit_transform(y)

cv_iris_log_model = sk_linear.LogisticRegression()
cv_iris_model_quality = sk_model_selection.cross_val_score(cv_iris_log_model, X, y, cv=4, scoring="accuracy")

print("Original model quality:")
# print(cv_iris_model_quality)
print(np.mean(cv_iris_model_quality))

pca_2_components = sk_decomposition.PCA(n_components=2)
principal_components = pca_2_components.fit_transform(X)

plt.scatter(principal_components[:, 0], principal_components[:, 1], c=y, cmap="prism")

pca_all_components = sk_decomposition.PCA()
pca_all_components.fit(X)

print("Explained variance ratio:")
print(pca_all_components.explained_variance_ratio_)

cv_iris_2_log_model = sk_linear.LogisticRegression()
cv_iris_2_model_quality = sk_model_selection.cross_val_score(cv_iris_2_log_model, principal_components, y, cv=4,
                                                             scoring="accuracy")

print("2 pc model quality:")
# print(cv_iris_2_model_quality)
print(np.mean(cv_iris_2_model_quality))

plt.figure()
components = list(range(1, pca_all_components.n_components_ + 1))
plt.plot(components, np.cumsum(pca_all_components.explained_variance_ratio_), marker="o")
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")
plt.ylim(0, 1.1)

plt.show()
