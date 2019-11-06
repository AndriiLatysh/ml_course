import pandas as pd
import sklearn.linear_model as lm


advertising_data = pd.read_csv("data/advertising.csv", index_col=0)
print(advertising_data)

ad_data = advertising_data[["TV", "radio", "newspaper"]]
sales_data = advertising_data[["sales"]]

# linear_regression = lm.Ridge()
# linear_regression = lm.Lasso()
linear_regression = lm.LinearRegression()
linear_regression.fit(ad_data, sales_data)

print(linear_regression.coef_)
print(linear_regression.intercept_)
