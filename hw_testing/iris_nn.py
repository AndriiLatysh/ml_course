import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler


iris_dataset = pd.read_csv("data/iris.csv")
print(iris_dataset)
print(set(iris_dataset['class']))

# iris_virginica = iris_dataset[iris_dataset["class"] == 'Iris-virginica']
# iris_versicolor = iris_dataset[iris_dataset["class"] == 'Iris-versicolor']
# iris_setosa = iris_dataset[iris_dataset["class"] == 'Iris-setosa']


X = np.array(iris_dataset[["sepal-length", "sepal-width", "petal-length", "petal-width"]]).reshape(-1, 4)
y = np.array(iris_dataset[["class"]]).reshape(-1, 1)

standart_scaler = StandardScaler()
X = standart_scaler.fit_transform(X)

dummy_encoding = OneHotEncoder(sparse=False, categories="auto")
y = dummy_encoding.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

print("Training.")

ann = MLPClassifier(hidden_layer_sizes=(6,), max_iter=1000000)
ann.fit(X_train, y_train)

print("Training finished.")

y_predicted = ann.predict(X_test)
y_probabilities = ann.predict_proba(X_test)[:, 1]

# max_grade = 151
# sepal_length_range = list(range(max_grade))
# sepal_width_range = list(range(max_grade))
# petal_length_range = list(range(max_grade))
# petal_width_range = list(range(max_grade))
# probability_level = np.empty([max_grade, max_grade, max_grade, max_grade])
# for x in sepal_length_range:
#     for y in sepal_width_range:
#         for z in petal_length_range:
#             for w in petal_width_range:
#                 prediction_point = standart_scaler.transform(np.array([x, y, z, w]).reshape(1, -1))
#                 probability_level[x, y, z, w] = ann.predict_proba(prediction_point)[:, 1]print("Training.")

# plt.contourf(probability_level, cmap="RdYlBu")

y_test_labels = dummy_encoding.inverse_transform(y_test)
y_predicted_labels = dummy_encoding.inverse_transform(y_predicted)

print(confusion_matrix(y_test_labels, y_predicted_labels))
print(classification_report(y_test_labels, y_predicted_labels))

# plt.show()