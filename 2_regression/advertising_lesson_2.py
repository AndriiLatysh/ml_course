import pandas as pd
import sklearn.linear_model as lm


def model_to_string(model, labels, precision=4):
    model_str = "{} = ".format(labels[0])
    for z in range(1, len(labels)):
        model_str += "{} * {} + ".format(round(model.coef_.flatten()[z - 1], 4), labels[z])
    model_str += "{}".format(round(model.intercept_[0], 4))
    return model_str


advertising_data = pd.read_csv("data/advertising.csv", index_col=0)
print(advertising_data)

ad_data = advertising_data[["TV", "radio", "newspaper"]]
sales_data = advertising_data[["sales"]]

linear_regression = lm.LinearRegression()
lasso_regression = lm.Lasso()
ridge_regression = lm.Ridge()

linear_regression.fit(ad_data, sales_data)
lasso_regression.fit(ad_data, sales_data)
ridge_regression.fit(ad_data, sales_data)

labels = ["sales", "TV", "radio", "newspaper"]

print("Linear regression.")
print(model_to_string(linear_regression, labels))
print()

print("Ridge regression.")
print(model_to_string(ridge_regression, labels))
print()

print("Lasso regression.")
print(model_to_string(lasso_regression, labels))
print()