import pandas as pd
import sklearn.linear_model as lm


def model_to_string(model, labels, precision=4):
    model_str = "{} = ".format(labels[-1])
    for z in range(len(labels) - 1):
        model_str += "{} * {} + ".format(round(model.coef_.flatten()[z], precision), labels[z])
    model_str += "{}".format(round(model.intercept_[0], precision))
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

labels = advertising_data.columns.values

print("Linear regression.")
print(model_to_string(linear_regression, labels))
print()

print("Ridge regression.")
print(model_to_string(ridge_regression, labels))
print()

print("Lasso regression.")
print(model_to_string(lasso_regression, labels))
print()
