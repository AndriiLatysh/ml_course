import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.metrics as sm


# def visualize_single_variable_regression(x, y):
#     plt.scatter(x, y)
#
#     linear_regression = lm.LinearRegression()
#     linear_regression.fit(x, y)
#
#     model = linear_regression.predict(x)
#
#     plt.plot(x, model, color="r")
#
#     plt.show()


advertising_data = pd.read_csv("advertising.csv", index_col=0)
print(advertising_data)

# visualize_single_variable_regression(advertising_data[["TV"]], advertising_data[["sales"]])
# visualize_single_variable_regression(advertising_data[["radio"]], advertising_data[["sales"]])
# visualize_single_variable_regression(advertising_data[["newspaper"]], advertising_data[["sales"]])

ad_data = np.array(advertising_data[["TV", "radio", "newspaper"]])
sales_data = np.array(advertising_data[["sales"]])

linear_regression = lm.LinearRegression()
linear_regression.fit(ad_data, sales_data)

print(linear_regression.coef_)
print(linear_regression.intercept_)

modeled_sales = linear_regression.predict(ad_data)

print(sm.mean_squared_error(sales_data, modeled_sales))
print(math.sqrt(sm.mean_squared_error(sales_data, modeled_sales)))
