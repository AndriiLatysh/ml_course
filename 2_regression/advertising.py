import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn


def visualize_single_variable_regression(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)

    plt.scatter(x, y)

    linear_model = lm.LinearRegression()
    linear_model.fit(x, y)

    model = linear_model.predict(x)

    plt.plot(x, model, color="r")

    plt.show()


def model_to_string(model, labels):
    result_string = "{} = ".format(labels[0])
    for z in range(1, len(labels)):
        result_string += "{} * {} + ".format(model.coef_[0][z - 1], labels[z])
    result_string += "{}".format(model.intercept_[0])
    return result_string


def train_linear_model(X, y):
    trained_model = lm.LinearRegression()
    trained_model.fit(X, y)
    return trained_model


def get_MSE_for_linear_model(trained_model, X, y_true):
    y_predicted = trained_model.predict(X)
    MSE = sklearn.metrics.mean_squared_error(y_true, y_predicted)
    return MSE


advertising_data = pd.read_csv("data/advertising.csv", index_col=0)
print(advertising_data)

# visualize_single_variable_regression(advertising_data["TV"], advertising_data["sales"])
# visualize_single_variable_regression(advertising_data["radio"], advertising_data["sales"])
# visualize_single_variable_regression(advertising_data["newspaper"], advertising_data["sales"])

values = advertising_data.values
min_max_scaler = sklearn.preprocessing.MinMaxScaler()
values_scaled = min_max_scaler.fit_transform(values)
advertising_data = pd.DataFrame(values_scaled, columns=advertising_data.columns.values, index=advertising_data.index)
print(advertising_data)

ad_data = np.array(advertising_data[["TV", "radio", "newspaper"]]).reshape(-1, 3)
sales_data = np.array(advertising_data[["sales"]]).reshape(-1, 1)

linear_model = lm.LinearRegression()

linear_model.fit(ad_data, sales_data)

print(model_to_string(linear_model, ["sales", "TV", "radio", "newspapers"]))

modeled_sales = linear_model.predict(ad_data)

MSE = sklearn.metrics.mean_squared_error(sales_data, modeled_sales)
print("MSE = {}".format(MSE))

print("Cross validated models.")

X_train, X_test, y_train, y_test = ms.train_test_split(ad_data, sales_data)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

cv_linear_model = lm.LinearRegression()
cv_linear_model.fit(X_train, y_train)

print(model_to_string(cv_linear_model, ["sales", "TV", "radio", "newspapers"]))

cv_modeled_sales = cv_linear_model.predict(X_test)

cv_MSE = sklearn.metrics.mean_squared_error(y_test, cv_modeled_sales)
print("MSE = {}".format(cv_MSE))

print("Removing {}.".format("newspaper"))

X_train_no_newspaper = np.array([np.array(l[:-1]) for l in X_train])
# print(X_test)
X_test_no_newspaper = np.array([np.array(l[:-1]) for l in X_test])
# print(X_test)

linear_model_TV_radio = train_linear_model(X_train_no_newspaper, y_train)
print(model_to_string(linear_model_TV_radio, ["sales", "TV", "radio"]))
MSE_TV_radio = get_MSE_for_linear_model(linear_model_TV_radio, X_test_no_newspaper, y_test)
print("MSE = {}".format(MSE_TV_radio))

print("Removing {}.".format("radio"))

X_train_no_radio = np.array([np.delete(l, 1) for l in X_train])
# print(X_test)
X_test_no_radio = np.array([np.delete(l, 1) for l in X_test])
# print(X_test)

linear_model_TV_newspapers = train_linear_model(X_train_no_radio, y_train)
print(model_to_string(linear_model_TV_newspapers, ["sales", "TV", "newspapers"]))
MSE_TV_newspapers = get_MSE_for_linear_model(linear_model_TV_newspapers, X_test_no_radio, y_test)
print("MSE = {}".format(MSE_TV_newspapers))

print("Removing {}.".format("TV"))

X_train_no_TV = np.array([np.delete(l, 0) for l in X_train])
# print(X_test)
X_test_no_TV = np.array([np.delete(l, 0) for l in X_test])
# print(X_test)

linear_model_radio_newspapers = train_linear_model(X_train_no_TV, y_train)
print(model_to_string(linear_model_radio_newspapers, ["sales", "radio", "newspapers"]))
MSE_radio_newspapers = get_MSE_for_linear_model(linear_model_radio_newspapers, X_test_no_TV, y_test)
print("MSE = {}".format(MSE_radio_newspapers))
