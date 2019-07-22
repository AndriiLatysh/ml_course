import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from joblib import dump, load


qualifies_by_double_grade = pd.read_csv("data/double_grade.csv")
print(qualifies_by_double_grade)

qualified_candidates = qualifies_by_double_grade[qualifies_by_double_grade["qualifies"] == 1]
unqualified_candidates = qualifies_by_double_grade[qualifies_by_double_grade["qualifies"] == 0]

plt.xlabel("technical_grade")
plt.ylabel("english_grade")
plt.scatter(qualified_candidates["technical_grade"], qualified_candidates["english_grade"], color="g")
plt.scatter(unqualified_candidates["technical_grade"], unqualified_candidates["english_grade"], color="r")

X = np.array(qualifies_by_double_grade[["technical_grade", "english_grade"]]).reshape(-1, 2)
y = np.array(qualifies_by_double_grade["qualifies"])

# X_train, X_test, y_train, y_test = ms.train_test_split(X, y)
#
# logistic_model = LogisticRegression(solver="lbfgs")
# logistic_model.fit(X_train, y_train)
# y_modeled = logistic_model.predict(X_test)
#
# logistic_confusion_matrix = metrics.logistic_confusion_matrix(y_test, y_modeled)
# print(logistic_confusion_matrix)

k_folds = ms.KFold(n_splits=4, shuffle=False)
confusion_matrix = np.array([[0, 0], [0, 0]])

for train_index, test_index in k_folds.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    logistic_model = LogisticRegression(solver="lbfgs")
    logistic_model.fit(X_train, y_train)
    y_modeled = logistic_model.predict(X_test)

    test_confusion_matrix = metrics.confusion_matrix(y_test, y_modeled)
    print(test_confusion_matrix)

    confusion_matrix += test_confusion_matrix

print("Confusion matrix:")
print(confusion_matrix)

final_model = LogisticRegression(solver="lbfgs")
final_model.fit(X, y)

dump(final_model, "models/qualification_by_double_grade_model.joblib")

# plt.show()
