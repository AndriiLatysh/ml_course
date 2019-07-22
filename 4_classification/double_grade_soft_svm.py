import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from sklearn import svm
from sklearn import metrics


def plot_model(svm_classifier):
    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # create grid to evaluate model
    plotting_step = 50
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


qualifies_by_double_grade = pd.read_csv("data/double_grade_reevaluated.csv")
print(qualifies_by_double_grade)

qualified_candidates = qualifies_by_double_grade[qualifies_by_double_grade["qualifies"] == 1]
unqualified_candidates = qualifies_by_double_grade[qualifies_by_double_grade["qualifies"] == 0]

plt.xlabel("technical_grade")
plt.ylabel("english_grade")
plt.scatter(qualified_candidates["technical_grade"], qualified_candidates["english_grade"], color="g")
plt.scatter(unqualified_candidates["technical_grade"], unqualified_candidates["english_grade"], color="r")

X = np.array(qualifies_by_double_grade[["technical_grade", "english_grade"]]).reshape(-1, 2)
y = np.array(qualifies_by_double_grade["qualifies"])

# k_folds = ms.KFold(n_splits=4, shuffle=False)
# confusion_matrix = np.array([[0, 0], [0, 0]])
#
# model_kernel = "linear"
# model_C = 1
#
# for train_index, test_index in k_folds.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
#     svm_classifier = svm.SVC(C=model_C, kernel=model_kernel)
#     svm_classifier.fit(X_train, y_train)
#     y_modeled = svm_classifier.predict(X_test)
#
#     test_confusion_matrix = metrics.confusion_matrix(y_test, y_modeled)
#     print(test_confusion_matrix)
#
#     confusion_matrix += test_confusion_matrix
#
# print("Confusion matrix:")
# print(confusion_matrix)
#
# svm_classifier = svm.SVC(C=model_C, kernel=model_kernel)
# svm_classifier.fit(X, y)
#
# plot_model(svm_classifier)

# Grid Search
# Parameter Grid
param_grid = {"kernel": ["rbf"], "C": [1e-3, 1e-2, 1e-1, 1, 10, 100], "gamma": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]}

# Make grid search classifier
clf_grid = ms.GridSearchCV(svm.SVC(), param_grid, cv=5, iid=False)

# Train the classifier
clf_grid.fit(X, y)

# clf = grid.best_estimator_()
print("Best Parameters:\n", clf_grid.best_params_)

plot_model(clf_grid.best_estimator_)

y_modeled = clf_grid.predict(X)

confusion_matrix = metrics.confusion_matrix(y, y_modeled)

print(confusion_matrix)

print("Accuracy:", metrics.accuracy_score(y, y_modeled))
print("Error Rate:", 1 - metrics.accuracy_score(y, y_modeled))
print("Precision:", metrics.precision_score(y, y_modeled))
print("Recall:", metrics.recall_score(y, y_modeled))

plt.show()
