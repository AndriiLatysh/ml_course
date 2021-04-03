import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as sk_model_selection
import sklearn.metrics as sk_metrics
import sklearn.svm as sk_svm
import double_grade_svm_utility


qualifies_double_grade_df = pd.read_csv("data/double_grade.csv")

double_grade_svm_utility.plot_values(qualifies_double_grade_df)

X = qualifies_double_grade_df[["technical_grade", "english_grade"]]
y = qualifies_double_grade_df["qualifies"]

cv_svm_soft_linear_classifier = sk_svm.SVC(kernel="linear")
cv_svm_soft_linear_predictions = sk_model_selection.cross_val_predict(cv_svm_soft_linear_classifier, X, y, cv=4)

cv_confusion_matrix = sk_metrics.confusion_matrix(y, cv_svm_soft_linear_predictions)
print(cv_confusion_matrix)

svm_soft_linear_classifier = sk_svm.SVC(kernel="linear")
svm_soft_linear_classifier.fit(X, y)

print(svm_soft_linear_classifier.coef_)
print(svm_soft_linear_classifier.intercept_)

double_grade_svm_utility.plot_model(svm_soft_linear_classifier)

plt.show()
