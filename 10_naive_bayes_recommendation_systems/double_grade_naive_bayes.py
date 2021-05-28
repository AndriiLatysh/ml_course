import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.naive_bayes as sk_naive_bayes
import sklearn.model_selection as sk_ms
import sklearn.metrics as sk_metrics
import double_grade_utility


qualifies_double_grade_df = pd.read_csv("data/double_grade_reevaluated.csv")

sns.pairplot(qualifies_double_grade_df, hue="qualifies")

X = qualifies_double_grade_df[["technical_grade", "english_grade"]]
y = qualifies_double_grade_df["qualifies"]

# k_folds = sk_ms.StratifiedKFold(n_splits=4, shuffle=True)

naive_bayes_model = sk_naive_bayes.GaussianNB()
cv_predictions = sk_ms.cross_val_predict(naive_bayes_model, X, y, cv=4)

confusion_matrix = sk_metrics.confusion_matrix(y, cv_predictions)
print(confusion_matrix)

naive_bayes_model.fit(X, y)

plt.figure()
double_grade_utility.plot_model(naive_bayes_model, qualifies_double_grade_df)

plt.show()
