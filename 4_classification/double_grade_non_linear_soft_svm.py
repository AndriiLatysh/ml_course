import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import sklearn.metrics as metrics
import sklearn.svm as svm


def plot_values(qualifies_double_grade_df):
    plt.xlabel("Technical grade")
    plt.ylabel("English grade")

    qualified_candidates = qualifies_double_grade_df[qualifies_double_grade_df["qualifies"] == 1]
    unqualified_candidates = qualifies_double_grade_df[qualifies_double_grade_df["qualifies"] == 0]

    plt.scatter(qualified_candidates["technical_grade"], qualified_candidates["english_grade"], color="g")
    plt.scatter(unqualified_candidates["technical_grade"], unqualified_candidates["english_grade"], color="r")


def plot_model(svm_classifier):
    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # create grid to evaluate model
    plotting_step = 100
    xx = np.linspace(xlim[0], xlim[1], plotting_step)
    yy = np.linspace(ylim[0], ylim[1], plotting_step)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = svm_classifier.decision_function(xy).reshape(XX.shape)
    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(svm_classifier.support_vectors_[:, 0], svm_classifier.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none')


def compare_variance_for_vectors(X):
    X = np.array(X)
    X_var = X.var()
    print("X.var(): {}".format(X_var))
    # X_mean = np.mean(X, axis=0)
    # X_var_manual = np.mean([np.dot((x - X_mean), (x - X_mean)) for x in X])/2
    X_flat = X.flatten()
    print(X_flat)
    X_var_manual = X_flat.var()
    print("Manual variance: {}".format(X_var_manual))


qualifies_double_grade_df = pd.read_csv("data/double_grade_reevaluated.csv")

plot_values(qualifies_double_grade_df)

X = qualifies_double_grade_df[["technical_grade", "english_grade"]]
y = qualifies_double_grade_df["qualifies"]

parameter_grid = {"kernel": ["rbf"], "C": [10 ** p for p in range(-3, 5)], "gamma": [10 ** p for p in range(-6, 2)]}
# parameter_grid = {"kernel": ["rbf"], "C": [10 ** p for p in range(-3, 5)], "gamma": [p * 1e-4 for p in range(1, 10)]}

grid_search = ms.GridSearchCV(svm.SVC(), param_grid=parameter_grid, cv=4)
grid_search.fit(X, y)

print(grid_search.best_params_)

modeled_qualification = grid_search.predict(X)
confusion_matrix = metrics.confusion_matrix(y, modeled_qualification)

print(confusion_matrix)

plot_model(grid_search.best_estimator_)

# smv_soft_rbf_classifier = svm.SVC(kernel="rbf", gamma="scale")
# smv_soft_rbf_classifier.fit(X, y)
#
# plot_model(smv_soft_rbf_classifier)

plt.show()
