import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

iris_dataset = pd.read_csv("data/iris.csv")

X = np.array(iris_dataset[["sepal-length", "sepal-width", "petal-length", "petal-width"]]).reshape(-1, 4)
y = np.array(iris_dataset["class"]).reshape(-1, 1)

dummy_encoder = preprocessing.OneHotEncoder(sparse=False, categories="auto")
y = dummy_encoder.fit_transform(y)

standard_scaler = preprocessing.StandardScaler()
X = standard_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

mlp = MLPClassifier(hidden_layer_sizes=(5, ), max_iter=5000)
mlp.fit(X_train, y_train)

y_predictions = mlp.predict(X_test)

y_test_labels = dummy_encoder.inverse_transform(y_test)
predictions = dummy_encoder.inverse_transform(y_predictions)

print(confusion_matrix(y_test_labels, predictions))
print(classification_report(y_test_labels, predictions))
