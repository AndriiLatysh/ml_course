import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as sk_preprocessing
import sklearn.decomposition as sk_decomposition


def plot_values(qualifies_double_grade_df):
    plt.plot()

    qualified_candidates = qualifies_double_grade_df[qualifies_double_grade_df["qualifies"] == 1]
    unqualified_candidates = qualifies_double_grade_df[qualifies_double_grade_df["qualifies"] == 0]

    plt.scatter(qualified_candidates.iloc[:, 0], qualified_candidates.iloc[:, 1], color="g")
    plt.scatter(unqualified_candidates.iloc[:, 0], unqualified_candidates.iloc[:, 1], color="r")


double_grade_df = pd.read_csv("data/double_grade_reevaluated.csv")
# plot_values(double_grade_df)

column_names = double_grade_df.columns.tolist()

X = double_grade_df[column_names[:-1]]
y = double_grade_df[column_names[-1]]

standard_scaler = sk_preprocessing.StandardScaler()
X = standard_scaler.fit_transform(X)

pca_all = sk_decomposition.KernelPCA(n_components=2, kernel="rbf", gamma=1e-6)
principal_components = pca_all.fit_transform(X)

double_grade_transformed_df = pd.DataFrame(data=principal_components, columns=["pc 1", "pc 2"])
double_grade_transformed_df = pd.concat([double_grade_transformed_df, y], axis=1)

plot_values(double_grade_transformed_df)

print(double_grade_transformed_df)

plt.show()
