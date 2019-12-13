import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def set_printing_options():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 200)


set_printing_options()

qualifies_by_double_grade = pd.read_csv("data/double_grade_reevaluated.csv")

qualified_candidates = qualifies_by_double_grade[qualifies_by_double_grade["qualifies"] == 1]
unqualified_candidates = qualifies_by_double_grade[qualifies_by_double_grade["qualifies"] == 0]

plt.xlabel("technical_grade")
plt.ylabel("english_grade")

X = np.array(qualifies_by_double_grade[["technical_grade", "english_grade"]]).reshape(-1, 2)
y = np.array(qualifies_by_double_grade["qualifies"])

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

ann_model = MLPClassifier(hidden_layer_sizes=(4), max_iter=1000000)
ann_model.fit(X_train, y_train)

y_predicted = ann_model.predict(X_test)

y_probabilities = ann_model.predict_proba(X_test)[:, 1]

max_grade = 101
english_grades_range = list(range(max_grade))
technical_grades_range = list(range(max_grade))
probability_level = np.empty([max_grade, max_grade])
for x in technical_grades_range:
    for y in english_grades_range:
        probability_level[x, y] = ann_model.predict_proba(np.array([x, y]).reshape(1, -1))[:, 1]

plt.contourf(probability_level, cmap="RdYlBu")

plt.scatter(qualified_candidates["technical_grade"], qualified_candidates["english_grade"], color="g")
plt.scatter(unqualified_candidates["technical_grade"], unqualified_candidates["english_grade"], color="r")

y_test_labels = y_test
y_predicted_labels = y_predicted

print(confusion_matrix(y_test_labels, y_predicted_labels))
print(classification_report(y_test_labels, y_predicted_labels))

plt.show()
