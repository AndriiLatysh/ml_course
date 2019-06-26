import numpy as np
import pandas as pd
import sklearn.model_selection as ms
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier


qualifies_by_double_grade = pd.read_csv("data/double_grade_reevaluated.csv")

qualified_candidates = qualifies_by_double_grade[qualifies_by_double_grade["qualifies"] == 1]
unqualified_candidates = qualifies_by_double_grade[qualifies_by_double_grade["qualifies"] == 0]

X = np.array(qualifies_by_double_grade[["technical_grade", "english_grade"]]).reshape(-1, 2)
y = np.array(qualifies_by_double_grade["qualifies"])

standard_scaler = preprocessing.StandardScaler()
X = standard_scaler.fit_transform(X)

k_folds = ms.KFold(n_splits=4, shuffle=True)

# classifier_model = SVC(kernel="rbf", gamma="scale", probability=True)
classifier_model = MLPClassifier(hidden_layer_sizes=(6, 4), max_iter=1000000)
# classifier_model = DecisionTreeClassifier()

results = model_selection.cross_val_score(classifier_model, X, y, cv=k_folds)

print("Average accuracy: {}".format(results.mean()))
