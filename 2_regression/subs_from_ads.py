import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm


subscribers_from_ads = pd.read_csv("data/subscribers_from_ads.csv")
print(subscribers_from_ads)

plt.scatter(subscribers_from_ads[["promotion_budget"]], subscribers_from_ads[["subscribers"]])

linear_regression = lm.LinearRegression()
promotion_budget = subscribers_from_ads[["promotion_budget"]]
number_of_subscribers = subscribers_from_ads[["subscribers"]]
linear_regression.fit(X=promotion_budget, y=number_of_subscribers)

print(linear_regression.coef_)
print(linear_regression.intercept_)

regression_line_points = linear_regression.predict(X=promotion_budget)

plt.plot(promotion_budget, regression_line_points)

plt.show()
