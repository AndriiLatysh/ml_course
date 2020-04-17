import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.svm as svm
import double_grade_svm_utility


qualifies_double_grade_df = pd.read_csv("data/double_grade_small.csv")

double_grade_svm_utility.plot_values(qualifies_double_grade_df)

X = qualifies_double_grade_df[["technical_grade", "english_grade"]]
y = qualifies_double_grade_df["qualifies"]

smv_hard_linear_classifier = svm.SVC(kernel="linear")
smv_hard_linear_classifier.fit(X, y)

double_grade_svm_utility.plot_model(smv_hard_linear_classifier)

plt.show()
