import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree as sk_trees
import sklearn.ensemble as sk_ensemble
import double_grade_utility


# qualifies_double_grade_df = pd.read_csv("data/double_grade_small.csv")
# qualifies_double_grade_df = pd.read_csv("data/double_grade.csv")
qualifies_double_grade_df = pd.read_csv("data/double_grade_reevaluated.csv")

X = qualifies_double_grade_df[["technical_grade", "english_grade"]]
y = qualifies_double_grade_df["qualifies"]

tree_classifier = sk_trees.DecisionTreeClassifier()
# tree_classifier = sk_ensemble.RandomForestClassifier(n_jobs=-1)

tree_classifier.fit(X, y)

double_grade_utility.plot_model(tree_classifier, qualifies_double_grade_df)

plt.show()
