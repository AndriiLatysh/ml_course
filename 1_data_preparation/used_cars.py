import pandas as pd


def set_printing_options():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)


set_printing_options()

used_cars = pd.read_csv("true_car_listings.csv")
used_cars_sample = used_cars.sample(n=30)
print(used_cars_sample.to_string(index=False))
