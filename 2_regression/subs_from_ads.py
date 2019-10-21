import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm


subscribers_from_ads = pd.read_csv("subscribers_from_ads.csv")
print(subscribers_from_ads)
plt.scatter(subscribers_from_ads["promotion_budget"], subscribers_from_ads["subscribers"])

linear_model = lm.LinearRegression()
promotion_budget = np.array(subscribers_from_ads["promotion_budget"]).reshape(-1, 1)
number_of_subscribers = np.array(subscribers_from_ads["subscribers"]).reshape(-1, 1)

linear_model.fit(promotion_budget, number_of_subscribers)

print(linear_model.coef_)
print(linear_model.intercept_)

model = linear_model.predict(promotion_budget)

plt.plot(promotion_budget, model)

plt.show()
