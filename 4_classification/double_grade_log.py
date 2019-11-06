import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.metrics as sm
import joblib


def plot_model(model):
    plt.xlabel("Technical grade")
    plt.ylabel("English grade")

    qualified_candidates = qualifies_double_grade_df[qualifies_double_grade_df["qualifies"] == 1]
    unqualified_candidates = qualifies_double_grade_df[qualifies_double_grade_df["qualifies"] == 0]

    max_grade = 101
    english_grades_range = list(range(max_grade))
    technical_grades_range = list(range(max_grade))
    probability_level = np.empty([max_grade, max_grade])
    for x in technical_grades_range:
        for y in english_grades_range:
            prediction_point = (np.array([x, y]).reshape(1, -1))
            probability_level[x, y] = model.predict_proba(prediction_point)[:, 1]

    plt.contourf(probability_level, cmap="rainbow")  # cmap="RdYlBu"/"binary"

    plt.scatter(qualified_candidates["technical_grade"], qualified_candidates["english_grade"], color="w")
    plt.scatter(unqualified_candidates["technical_grade"], unqualified_candidates["english_grade"], color="k")


qualifies_double_grade_df = pd.read_csv("data/double_grade.csv")

X = qualifies_double_grade_df[["technical_grade", "english_grade"]]
y = qualifies_double_grade_df["qualifies"]

number_of_folds = 4

cv_qualification_model = lm.LogisticRegression(solver="lbfgs")
cv_model_quality = ms.cross_val_score(cv_qualification_model, X, y, cv=number_of_folds, scoring="accuracy")
print(cv_model_quality)

prediction_model_quality = ms.cross_val_predict(cv_qualification_model, X, y, cv=number_of_folds)
cv_confusion_matrix = sm.confusion_matrix(y, prediction_model_quality)
print(cv_confusion_matrix)

qualification_model = lm.LogisticRegression(solver="lbfgs")
qualification_model.fit(X, y)

modeled_qualification_probabilities = qualification_model.predict_proba(X)[:, 1]
qualifies_double_grade_df["modeled probability"] = modeled_qualification_probabilities

pd.set_option("display.max_rows", None)
print(qualifies_double_grade_df.sort_values(by="modeled probability"))

print(qualification_model.coef_)
print(qualification_model.intercept_)

plot_model(qualification_model)

joblib.dump(qualification_model, "models/qualification_by_two_grades_model.joblib")

# predicted_qualification = qualification_model.predict(X)
# confusion_matrix = sm.confusion_matrix(y, predicted_qualification)
# print(confusion_matrix)

plt.show()
