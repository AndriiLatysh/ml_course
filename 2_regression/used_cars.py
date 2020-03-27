import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

used_cars_df = pd.read_csv("data/true_car_listings.csv")

print(len(used_cars_df))

print(used_cars_df[["Year"]].max())

used_cars_df[["Age"]] = used_cars_df[["Year"]].max() - used_cars_df[["Year"]]

print(used_cars_df.head())

# plt.scatter(x=used_cars_df[["Age"]], y=used_cars_df[["Price"]])

model_list = used_cars_df[["Model", "Vin"]].groupby("Model").count().sort_values(by="Vin", ascending=False)

print(model_list)

selected_model_df = used_cars_df[used_cars_df["Model"] == "Civic"]

print(len(selected_model_df))

print(selected_model_df)

# threshold = 30000
# print("Values below threshold of {}: {}".format(threshold,
#                                                 len(selected_model_df[selected_model_df["Price"] > threshold])))

plt.scatter(x=selected_model_df[["Age"]], y=selected_model_df[["Price"]])

price_by_age_regression = linear_model.LinearRegression()

price_by_age_regression.fit(X=selected_model_df[["Age"]], y=selected_model_df[["Price"]])

print(price_by_age_regression.coef_, price_by_age_regression.intercept_)

age_range = [[selected_model_df["Age"].min()], [selected_model_df["Age"].max()]]

predicted_price_by_age = price_by_age_regression.predict(X=age_range)

plt.plot(age_range, predicted_price_by_age)

plt.show()
