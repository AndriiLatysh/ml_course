import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
import sklearn.model_selection as ms
from sklearn import metrics


def model_linear_vs_logistic_regression (model_data):
    model_data.sort_values(by=model_data.columns[0], inplace=True)

    data_1 = model_data[model_data[model_data.columns[1]] == 1]
    data_0 = model_data[model_data[model_data.columns[1]] == 0]

    plt.scatter(data_1[data_1.columns[0]], data_1[data_1.columns[1]], color='g')
    plt.scatter(data_0[data_1.columns[0]], data_0[data_1.columns[1]], color='r')

    X = np.array(model_data[model_data.columns[0]]).reshape(-1, 1)
    y = np.array(model_data[model_data.columns[1]])

    X_train, X_test, y_train, y_test = ms.train_test_split(X, y)

    test_df = pd.DataFrame(data={"X_test": X_test.ravel(), "y_test": y_test.ravel()})
    print(test_df)
    test_df.sort_values(by="X_test", inplace=True)
    X_test = np.array(test_df["X_test"]).reshape(-1, 1)
    y_test = np.array(test_df["y_test"]).reshape(-1, 1)

    # 1. Передбачення.'
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_prediction_probability = linear_model.predict(X_test)
    linear_modeled_values = [int(round(v)) for v in linear_prediction_probability]

    logistic_model = LogisticRegression(solver='lbfgs')
    logistic_model.fit(X_train, y_train)
    logistic_prediction_probabilities = logistic_model.predict_proba(X_test)[:, 1]
    logistic_modeled_values = logistic_model.predict(X_test)


    print('3. Матрицz збентеження.')
    print('Linear model.')
    linear_confusion_matrix = metrics.confusion_matrix(y_test, linear_modeled_values)
    print(linear_confusion_matrix)
    print('Logistic model.')
    logistic_confusion_matrix = metrics.confusion_matrix(y_test, logistic_modeled_values)
    print(logistic_confusion_matrix)
    print()

    print('4. Точність, Рівень помилки, Влучність, Чутливість.')
    print('Linear model.')
    print("Accuracy:", metrics.accuracy_score(y_test, linear_modeled_values))
    print("Error Rate:", 1 - metrics.accuracy_score(y_test, linear_modeled_values))
    print("Precision:", metrics.precision_score(y_test, linear_modeled_values))
    print("Recall:", metrics.recall_score(y_test, linear_modeled_values))
    print('Logistic model.')
    print("Accuracy:", metrics.accuracy_score(y_test, logistic_modeled_values))
    print("Error Rate:", 1 - metrics.accuracy_score(y_test, logistic_modeled_values))
    print("Precision:", metrics.precision_score(y_test, logistic_modeled_values))
    print("Recall:", metrics.recall_score(y_test, logistic_modeled_values))
    print()

    # 5. Побудовa передбачень і ймовірностей.
    plt.plot(X_test, linear_modeled_values, color="y")
    plt.plot(X_test, linear_prediction_probability, color="m")
    plt.plot(X_test, logistic_modeled_values, color="b")
    plt.plot(X_test, logistic_prediction_probabilities, color="c")
    plt.show()

print('Порівняння лінійної й логістичної регресії в задачах класифікації на наборах даних single_grade.csv')
single_grade = pd.read_csv('data/single_grade.csv')
model_linear_vs_logistic_regression(single_grade)
print()
print()

print('Порівняння лінійної й логістичної регресії в задачах класифікації на наборах даних linear_vs_logistic.csv')
linear_vs_logistic = pd.read_csv('data/linear_vs_logistic.csv')
model_linear_vs_logistic_regression(linear_vs_logistic)