import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree as sk_trees
import sklearn.ensemble as sk_ensemble


def plot_model(model, qualifies_double_grade_df):
    plt.xlabel("Technical grade")
    plt.ylabel("English grade")

    qualified_candidates = qualifies_double_grade_df[qualifies_double_grade_df["qualifies"] == 1]
    unqualified_candidates = qualifies_double_grade_df[qualifies_double_grade_df["qualifies"] == 0]

    max_grade = 101
    probability_level = np.empty([max_grade, max_grade])
    for technical_grade in range(max_grade):
        for english_grade in range(max_grade):
            prediction_point = [[technical_grade, english_grade]]
            probability_level[technical_grade, english_grade] = model.predict_proba(prediction_point)[:, 1]

    plt.contourf(probability_level, cmap="rainbow")  # cmap="RdYlBu"/"binary"

    plt.scatter(qualified_candidates["technical_grade"], qualified_candidates["english_grade"], color="w")
    plt.scatter(unqualified_candidates["technical_grade"], unqualified_candidates["english_grade"], color="k")


qualifies_double_grade_df = pd.read_csv("data/double_grade_reevaluated.csv")

X = qualifies_double_grade_df[["technical_grade", "english_grade"]]
y = qualifies_double_grade_df["qualifies"]

# tree_classifier = sk_trees.DecisionTreeClassifier()
tree_classifier = sk_ensemble.RandomForestClassifier(n_estimators=10, max_features=None)
tree_classifier.fit(X, y)

plot_model(tree_classifier, qualifies_double_grade_df)

plt.show()
