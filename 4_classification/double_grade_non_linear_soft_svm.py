import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as sk_model_selection
import sklearn.metrics as sk_metrics
import sklearn.svm as sk_svm
import double_grade_svm_utility


qualifies_double_grade_df = pd.read_csv("data/double_grade_reevaluated.csv")

double_grade_svm_utility.plot_values(qualifies_double_grade_df)

X = qualifies_double_grade_df[["technical_grade", "english_grade"]]
y = qualifies_double_grade_df["qualifies"]

# double_grade_svm_utility.compare_variance_for_vectors(X)
#
# svm_soft_non_linear_classifier = svm.SVC(kernel="rbf")
# # svm_soft_non_linear_classifier = svm.SVC(kernel="rbf", gamma="auto")
# svm_soft_non_linear_classifier.fit(X, y)
#
# double_grade_svm_utility.plot_model(svm_soft_non_linear_classifier)

parameter_grid = {"kernel": ["rbf"], "C": [10 ** p for p in range(-2, 6)], "gamma": [10 ** p for p in range(-6, 2)]}
# parameter_grid = {"kernel": ["rbf"], "C": [10 ** p for p in range(1, 7)], "gamma": [p * 1e-5 for p in range(1, 10)]}

grid_search = sk_model_selection.GridSearchCV(sk_svm.SVC(), param_grid=parameter_grid, cv=4)
grid_search.fit(X, y)

print(grid_search.best_params_)

modeled_qualification = grid_search.predict(X)
confusion_matrix = sk_metrics.confusion_matrix(y, modeled_qualification)

print(confusion_matrix)

double_grade_svm_utility.plot_model(grid_search.best_estimator_)

plt.show()
